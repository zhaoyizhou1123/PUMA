#!/usr/bin/env python3
"""
Generate Sudoku puzzles of arbitrary box size n using a fast canonical construction
with optional uniqueness verification via Google OR-Tools CP-SAT.

Board generation strategy
-------------------------
Instead of backtracking-solving an empty grid (py-sudoku takes ~30 s for n=5),
we build a valid complete board in O(n^4) time using the formula:

    board[i][j] = (n*(i % n) + i//n + j) % n^2 + 1

This satisfies all row, column, and box constraints for any n.
We then diversify with symmetry-preserving shuffles:
  - permute rows within each band
  - permute columns within each stack
  - permute entire bands / stacks
  - relabel digit symbols

Each transform preserves Sudoku validity, giving astronomically many
distinct boards from a single seed.  Generation time: ~0.6 ms for n=5.

Uniqueness checking (--check-uniqueness)
-----------------------------------------
By default puzzles may have multiple solutions (fine for masked diffusion
training data).  Pass --check-uniqueness to verify each puzzle has exactly
one solution using OR-Tools CP-SAT.  This adds ~2 s per puzzle for n=5 and
will significantly reduce throughput.

Output format per row: [k, r0, c0, v0, s0, r1, c1, v1, s1, ...]
  - k        : number of givens
  - quad_i   : (row, col, value, strategy=0), 4 ints per cell
  - first k quads  : given cells
  - next n^4-k quads : empty cells (solution values)
  - total  : 1 + 4*n^4 ints per puzzle

Saves:
  <output_dir>/sudoku-train-data.npy   shape (num_train, 1+4*n^4)
  <output_dir>/sudoku-test-data.npy    shape (num_test,  1+4*n^4)

Usage:
  python data/generate_sudoku_or_tools.py --n 5 --num-train 50000 --num-test 5000
  python data/generate_sudoku_or_tools.py --n 5 --num-train 1000  --check-uniqueness
"""

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

try:
    from ortools.sat.python import cp_model as _cp_model  # noqa: F401 – verify install
except ImportError:
    sys.exit("ortools not installed.  Run: pip install ortools")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Fast board generation via canonical construction + symmetry shuffling
# ---------------------------------------------------------------------------

def _canonical_solution(n: int, seed: int) -> list[list[int]]:
    """
    Return a complete, valid n^2 x n^2 Sudoku board in ~0.6 ms for n=5.

    The base formula  board[i][j] = (n*(i%n) + i//n + j) % n^2 + 1  satisfies
    all row, column, and box constraints.  Six symmetry-preserving shuffles
    then produce a diverse random board:
      1. rows within each band
      2. columns within each stack
      3. entire bands
      4. entire stacks
      5. digit relabeling
    """
    grid = n * n
    rng = random.Random(seed)

    board = [
        [(n * (i % n) + i // n + j) % grid + 1 for j in range(grid)]
        for i in range(grid)
    ]

    # 1. Permute rows within each band
    for band in range(n):
        s = band * n
        chunk = board[s : s + n]
        rng.shuffle(chunk)
        board[s : s + n] = chunk

    # 2. Permute columns within each stack
    for stack in range(n):
        s = stack * n
        cols = [[board[i][s + c] for i in range(grid)] for c in range(n)]
        rng.shuffle(cols)
        for c, col in enumerate(cols):
            for i in range(grid):
                board[i][s + c] = col[i]

    # 3. Permute entire bands
    band_order = list(range(n))
    rng.shuffle(band_order)
    board = [row for b in band_order for row in board[b * n : (b + 1) * n]]

    # 4. Permute entire stacks
    stack_order = list(range(n))
    rng.shuffle(stack_order)
    board = [
        [board[i][st * n + c] for st in stack_order for c in range(n)]
        for i in range(grid)
    ]

    # 5. Relabel digits
    digits = list(range(1, grid + 1))
    rng.shuffle(digits)
    mapping = {k + 1: digits[k] for k in range(grid)}
    return [[mapping[v] for v in row] for row in board]


# ---------------------------------------------------------------------------
# OR-Tools uniqueness checker
# ---------------------------------------------------------------------------

def _build_sudoku_model(n: int, prompt_board: list[list]):
    """
    Build a CP-SAT model for the given partial puzzle.
    Returns (model, cells).  Given cells are fixed; empty cells are free.
    """
    from ortools.sat.python import cp_model

    grid = n * n
    model = cp_model.CpModel()
    cells = [
        [model.NewIntVar(1, grid, f"c_{i}_{j}") for j in range(grid)]
        for i in range(grid)
    ]

    for i in range(grid):
        model.AddAllDifferent(cells[i])
    for j in range(grid):
        model.AddAllDifferent([cells[i][j] for i in range(grid)])
    for bi in range(n):
        for bj in range(n):
            model.AddAllDifferent(
                [cells[bi * n + r][bj * n + c] for r in range(n) for c in range(n)]
            )

    for i in range(grid):
        for j in range(grid):
            v = prompt_board[i][j]
            if v is not None:
                model.Add(cells[i][j] == v)

    return model, cells


def _is_unique_solution(n: int, prompt_board: list[list], known_solution: list[list]) -> bool:
    """
    Return True iff the puzzle has exactly one solution.

    Strategy: find the first solution (using known_solution as hints so it's
    fast), then add a constraint that forces at least one empty cell to differ
    and try again.  Unique iff the second solve is INFEASIBLE.
    """
    from ortools.sat.python import cp_model

    grid = n * n
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 60.0

    # --- First solve: confirm the known solution is reachable ---
    model1, cells1 = _build_sudoku_model(n, prompt_board)
    for i in range(grid):
        for j in range(grid):
            model1.AddHint(cells1[i][j], known_solution[i][j])

    status1 = solver.Solve(model1)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return False  # puzzle has no solution — shouldn't happen

    sol1 = [[solver.Value(cells1[i][j]) for j in range(grid)] for i in range(grid)]

    # --- Second solve: find a solution different from sol1 ---
    model2, cells2 = _build_sudoku_model(n, prompt_board)

    # Constraint: at least one empty cell must differ from sol1
    diff_flags = []
    for i in range(grid):
        for j in range(grid):
            if prompt_board[i][j] is None:
                b = model2.NewBoolVar(f"d_{i}_{j}")
                model2.Add(cells2[i][j] != sol1[i][j]).OnlyEnforceIf(b)
                model2.Add(cells2[i][j] == sol1[i][j]).OnlyEnforceIf(b.Not())
                diff_flags.append(b)

    if not diff_flags:
        return True  # fully filled puzzle — trivially unique

    model2.AddBoolOr(diff_flags)

    status2 = solver.Solve(model2)
    return status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ---------------------------------------------------------------------------
# Worker (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _worker(args):
    """Generate one puzzle+solution pair.  Returns list of 1+4*n^4 ints or None."""
    n, difficulty_min, difficulty_max, seed, check_uniqueness = args
    try:
        grid = n * n
        rng = random.Random(seed)

        # Step 1: fast complete board
        solution = _canonical_solution(n, seed)

        # Step 2: randomly remove cells
        difficulty = rng.uniform(difficulty_min, difficulty_max)
        n_remove = int(difficulty * grid * grid)
        indices = list(range(grid * grid))
        rng.shuffle(indices)

        prompt_board = [row[:] for row in solution]
        for idx in indices[:n_remove]:
            prompt_board[idx // grid][idx % grid] = None

        # Step 3: optional uniqueness check via OR-Tools
        if check_uniqueness:
            if not _is_unique_solution(n, prompt_board, solution):
                return None

        # Sanity checks
        answer_flat = [v for row in solution for v in row]
        if len(answer_flat) != grid * grid or 0 in answer_flat:
            return None
        if any(v < 1 or v > grid for v in answer_flat):
            return None

        # Build quad format: [k, r0, c0, v0, s0, ...] — givens first, then empties
        givens = []
        empties = []
        for i in range(grid):
            for j in range(grid):
                pv = prompt_board[i][j]
                sv = solution[i][j]
                if pv is not None:
                    givens.append((i, j, pv, 0))
                else:
                    empties.append((i, j, sv, 0))

        k = len(givens)
        quads = givens + empties
        return [k] + [x for quad in quads for x in quad]

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

def generate(
    n: int,
    count: int,
    difficulty_min: float,
    difficulty_max: float,
    base_seed: int,
    num_workers: int,
    desc: str = "",
    check_uniqueness: bool = False,
) -> np.ndarray:
    """Generate `count` valid puzzles.  Returns array of shape (count, 1+4*n^4)."""
    rng = random.Random(base_seed)
    results = []
    t_start = time.time()
    last_logged = 0

    label = desc or f"n={n}"

    def _log(msg: str) -> None:
        if HAS_TQDM:
            tqdm.write(msg)
        else:
            print(msg, flush=True)

    progress = tqdm(total=count, desc=label) if HAS_TQDM else None

    def _make_args(k: int):
        return [
            (
                n,
                difficulty_min,
                difficulty_max,
                rng.randint(0, 2**31 - 1),
                check_uniqueness,
            )
            for _ in range(k)
        ]

    with mp.Pool(num_workers) as pool:
        while len(results) < count:
            needed = count - len(results)
            batch = max(needed + 64, int(needed * 1.05), num_workers * 4)
            args = _make_args(batch)

            for item in pool.imap_unordered(_worker, args, chunksize=8):
                if item is not None and len(results) < count:
                    results.append(item)
                    if progress:
                        progress.update(1)

                    milestone = len(results) // 1_000
                    if milestone > last_logged:
                        last_logged = milestone
                        elapsed = time.time() - t_start
                        rate = len(results) / elapsed if elapsed > 0 else 0.0
                        eta = (count - len(results)) / rate if rate > 0 else float("inf")
                        _log(
                            f"[{label}  {len(results):>{len(str(count))}}/{count}]  "
                            f"elapsed: {elapsed:6.1f}s   "
                            f"rate: {rate:6.1f} puz/s   "
                            f"ETA: {eta:6.1f}s"
                        )

    if progress:
        progress.close()

    total_elapsed = time.time() - t_start
    overall_rate = count / total_elapsed if total_elapsed > 0 else 0.0
    _log(
        f"[{label}] Done: {count} puzzles in {total_elapsed:.1f}s "
        f"({overall_rate:.1f} puz/s)"
    )

    return np.array(results[:count], dtype=np.int32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Sudoku puzzles of arbitrary box size n.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="Box size: 3 → 9×9, 4 → 16×16, 5 → 25×25",
    )
    parser.add_argument("--num-train", type=int, default=0,
                        help="Number of training puzzles")
    parser.add_argument("--num-test",  type=int, default=0,
                        help="Number of test puzzles")
    parser.add_argument("--difficulty-min", type=float, default=0.5,
                        help="Min fraction of cells removed (0–1)")
    parser.add_argument("--difficulty-max", type=float, default=0.75,
                        help="Max fraction of cells removed (0–1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/sudoku_n{n}/)")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                        help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (test split uses seed+1)")
    parser.add_argument(
        "--check-uniqueness", action="store_true", default=False,
        help=(
            "Verify each puzzle has exactly one solution using OR-Tools CP-SAT. "
            "Much slower (~2 s/puzzle for n=5); puzzles that fail are retried."
        ),
    )
    # Accepted for CLI compatibility with generate_sudoku.py; has no effect here
    # because uniqueness is never enforced by default.
    parser.add_argument("--allow-multi-solutions", action="store_true", default=False,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.n < 2:
        parser.error("--n must be >= 2")
    if not (0.0 < args.difficulty_min <= args.difficulty_max < 1.0):
        parser.error("Need 0 < difficulty-min <= difficulty-max < 1")

    out_dir = args.output_dir or f"data/sudoku_n{args.n}"
    os.makedirs(out_dir, exist_ok=True)

    grid    = args.n * args.n
    seq_len = 1 + 4 * grid * grid

    print(f"n={args.n}  ({grid}×{grid} grid, {grid*grid} cells, seq_len={seq_len})")
    print(f"difficulty : [{args.difficulty_min}, {args.difficulty_max}]")
    print(f"uniqueness : {'enforced via OR-Tools CP-SAT' if args.check_uniqueness else 'not checked (fast)'}")
    print(f"workers    : {args.workers}")
    print(f"output     : {out_dir}/")
    print()

    if args.num_train > 0:
        train = generate(
            args.n, args.num_train,
            args.difficulty_min, args.difficulty_max,
            args.seed, args.workers,
            desc="train",
            check_uniqueness=args.check_uniqueness,
        )
        train_path = os.path.join(out_dir, "sudoku-train-data.npy")
        np.save(train_path, train)
        print(f"Saved  {train_path}  {train.shape}")

    if args.num_test > 0:
        test = generate(
            args.n, args.num_test,
            args.difficulty_min, args.difficulty_max,
            args.seed + 1, args.workers,
            desc="test ",
            check_uniqueness=args.check_uniqueness,
        )
        test_path = os.path.join(out_dir, "sudoku-test-data.npy")
        np.save(test_path, test)
        print(f"Saved  {test_path}  {test.shape}")


if __name__ == "__main__":
    main()
