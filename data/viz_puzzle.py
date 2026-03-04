#!/usr/bin/env python3
"""Visualize the first puzzle from a generated .npy file."""

import sys
import numpy as np


def fmt_grid(grid, n):
    side = n * n
    lines = []
    for i in range(side):
        row_str = ""
        for j in range(side):
            v = grid[i, j]
            if n <= 3:
                cell = str(v) if v else "."
                row_str += cell + " "
            else:
                cell = f"{v:2d}" if v else " ."
                row_str += cell + " "
            if (j + 1) % n == 0 and j < side - 1:
                row_str += "| "
        lines.append(row_str)
        if (i + 1) % n == 0 and i < side - 1:
            sep_width = side * (2 if n <= 3 else 3) + (n - 1) * 2
            lines.append("-" * sep_width)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_puzzle.py <path.npy> [n]")
        print("  n: box size (default: auto-detect from seq_len)")
        sys.exit(1)

    path = sys.argv[1]
    data = np.load(path)
    print(f"File: {path}  shape={data.shape}\n")

    row = data[0]
    seq_len = len(row)

    # Auto-detect n: seq_len = 1 + 4*n^4
    if len(sys.argv) >= 3:
        n = int(sys.argv[2])
    else:
        n = None
        for candidate in range(2, 10):
            if 1 + 4 * candidate**4 == seq_len:
                n = candidate
                break
        if n is None:
            print(f"Could not auto-detect n from seq_len={seq_len}. Pass n as second arg.")
            sys.exit(1)

    side = n * n
    k = int(row[0])
    quads = row[1:].reshape(-1, 4)  # shape (side*side, 4): (row, col, val, strategy)

    prompt = np.zeros((side, side), dtype=int)
    answer = np.zeros((side, side), dtype=int)
    for r, c, v, _ in quads[:k]:   # givens
        prompt[r, c] = v
        answer[r, c] = v
    for r, c, v, _ in quads[k:]:   # empties
        answer[r, c] = v

    print("PROMPT:")
    print(fmt_grid(prompt, n))
    print()
    print("SOLUTION:")
    print(fmt_grid(answer, n))
    print(f"\nGivens: {k}, Empty: {side*side - k}")


if __name__ == "__main__":
    main()
