#!/usr/bin/env python3
"""
Generate Sudoku puzzles of arbitrary box size n using py-sudoku.

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
  python data/generate_sudoku_morphing.py --n 4 --num-train 100000 --num-test 10000
  python data/generate_sudoku_morphing.py --n 5 --num-train 50000  --num-test 5000 --workers 16
"""

import argparse
import multiprocessing as mp
import os
import random
import signal
import sys
import time

import numpy as np

# Remove the script's own directory from sys.path so that `import sudoku`
# resolves to the py-sudoku package rather than the local data/sudoku.py.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

try:
    from sudoku import Sudoku
except ImportError:
    sys.exit("py-sudoku not installed.  Run: pip install py-sudoku")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Worker (must be module-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _TimeoutError()

def _worker(args):
    """
    Generate one puzzle+solution pair.

    Returns a list of 1+4*n^4 ints on success, or None on any failure/timeout.
    """
    n, difficulty, seed, allow_multi_solutions, timeout_sec = args
    try:
        grid = n * n  # side length of the full board (e.g. 16 for n=4)

        if timeout_sec > 0:
            _old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_sec)

        try:
            if allow_multi_solutions:
                # Fast path: generate a complete solution, then remove cells without
                # any uniqueness check.  py-sudoku's difficulty() always runs
                # has_multiple_solutions() internally, so we bypass it entirely.
                solution = Sudoku(n, seed=seed).solve(assert_solvable=True)
                solution_board = solution.board

                rng = random.Random(seed)
                indices = list(range(grid * grid))
                rng.shuffle(indices)
                n_remove = int(difficulty * grid * grid)
                prompt_board = [row[:] for row in solution_board]
                for idx in indices[:n_remove]:
                    prompt_board[idx // grid][idx % grid] = None
            else:
                puzzle = Sudoku(n, seed=seed).difficulty(difficulty)

                # py-sudoku sets difficulty=-3 when it cannot guarantee a unique solution
                if getattr(puzzle, "difficulty", None) == -3:
                    return None

                solution = puzzle.solve(assert_solvable=True)
                prompt_board = puzzle.board
                solution_board = solution.board

        except _TimeoutError:
            return None
        finally:
            if timeout_sec > 0:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, _old_handler)

        # Sanity checks
        answer_flat = [int(v) for row in solution_board for v in row]
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
                sv = int(solution_board[i][j])
                if pv is not None:
                    givens.append((i, j, int(pv), 0))
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
    allow_multi_solutions: bool = False,
    timeout_sec: float = 0.0,
) -> np.ndarray:
    """
    Generate `count` valid puzzles.

    Returns numpy array of shape (count, 1+4*n^4), dtype int32.
    """
    rng = random.Random(base_seed)
    results = []
    t_start = time.time()
    last_logged = 0  # last milestone (in thousands) that was printed

    label = desc or f"n={n}"

    def _log(msg: str) -> None:
        if HAS_TQDM:
            tqdm.write(msg)
        else:
            print(msg, flush=True)

    progress = tqdm(total=count, desc=label) if HAS_TQDM else None

    def _make_args(k: int):
        return [
            (n, rng.uniform(difficulty_min, difficulty_max), rng.randint(0, 2**31 - 1), allow_multi_solutions, timeout_sec)
            for _ in range(k)
        ]

    with mp.Pool(num_workers) as pool:
        while len(results) < count:
            needed  = count - len(results)
            # Slightly over-generate to absorb failures; py-sudoku rarely fails,
            # but high difficulty or large n can occasionally trigger retries.
            batch   = max(needed + 64, int(needed * 1.05), num_workers * 4)
            args    = _make_args(batch)

            for item in pool.imap_unordered(_worker, args, chunksize=8):
                if item is not None:
                    results.append(item)
                    if progress:
                        progress.update(1)

                    # Log every 1 000 puzzles
                    milestone = len(results) // 1_000
                    if milestone > last_logged:
                        last_logged = milestone
                        elapsed = time.time() - t_start
                        rate    = len(results) / elapsed if elapsed > 0 else 0.0
                        eta     = (count - len(results)) / rate if rate > 0 else float("inf")
                        _log(
                            f"[{label}  {len(results):>{len(str(count))}}/{count}]  "
                            f"elapsed: {elapsed:6.1f}s   "
                            f"rate: {rate:6.1f} puz/s   "
                            f"ETA: {eta:6.1f}s"
                        )

                if len(results) >= count:
                    break

    if progress:
        progress.close()

    total_elapsed = time.time() - t_start
    overall_rate  = count / total_elapsed if total_elapsed > 0 else 0.0
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
                        help="Number of training puzzles to generate")
    parser.add_argument("--num-test",  type=int, default=0,
                        help="Number of test puzzles to generate")
    parser.add_argument("--difficulty-min", type=float, default=0.5,
                        help="Min fraction of cells removed (0–1)")
    parser.add_argument("--difficulty-max", type=float, default=0.75,
                        help="Max fraction of cells removed (0–1)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/sudoku_n{n}/)",
    )
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                        help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (test split uses seed+1)")
    parser.add_argument("--allow-multi-solutions", action="store_true", default=False,
                        help="Skip uniqueness check (much faster for n>=4; puzzles may have multiple solutions)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Per-puzzle timeout in seconds; slow seeds are skipped (0 to disable)")
    args = parser.parse_args()

    # Validate
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
    print(f"uniqueness : {'skipped (--allow-multi-solutions)' if args.allow_multi_solutions else 'enforced'}")
    print(f"workers    : {args.workers}")
    print(f"timeout    : {args.timeout}s per puzzle (0 = disabled)" if args.timeout > 0 else "timeout    : disabled")
    print(f"output     : {out_dir}/")
    print()

    # Train
    train = generate(
        args.n, args.num_train,
        args.difficulty_min, args.difficulty_max,
        args.seed, args.workers,
        desc="train",
        allow_multi_solutions=args.allow_multi_solutions,
        timeout_sec=args.timeout,
    )
    train_path = os.path.join(out_dir, "sudoku-train-data.npy")
    np.save(train_path, train)
    print(f"Saved  {train_path}  {train.shape}")

    # Test  (different seed to avoid overlap)
    test = generate(
        args.n, args.num_test,
        args.difficulty_min, args.difficulty_max,
        args.seed + 1, args.workers,
        desc="test ",
        allow_multi_solutions=args.allow_multi_solutions,
        timeout_sec=args.timeout,
    )
    test_path = os.path.join(out_dir, "sudoku-test-data.npy")
    np.save(test_path, test)
    print(f"Saved  {test_path}  {test.shape}")


if __name__ == "__main__":
    main()
