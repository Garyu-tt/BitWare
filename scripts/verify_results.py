#!/usr/bin/env python3
"""Check internal arithmetic of historical R04 CSVs; does not run CUDA."""
import csv
import math
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = ("results_2048.csv", "results_pack_2048.csv", "results_i16_2048.csv")
errors = []
count = 0
for name in files:
    with (root / "r04" / name).open(newline="") as fp:
        for row in csv.DictReader(fp):
            count += 1
            try:
                n = int(row["n"])
                k = int(row["k"])
                ms = float(row["ms"])
                tops = float(row["dense_equivalent_tops"])
                dense_ms = float(row["cublas_dense_ms"])
                speedup = float(row["speedup_vs_dense"])
                assert n > 0 and k > 0 and ms > 0 and dense_ms > 0
                assert row["validation"] == "PASS"
                assert math.isclose(tops, 2*n*n*k/(ms*1e-3)/1e12, rel_tol=0.0001)
                assert math.isclose(speedup, dense_ms/ms, rel_tol=0.0001)
            except (AssertionError, ValueError, KeyError, ZeroDivisionError) as exc:
                errors.append(f"{name}: {row!r}: {exc}")
if errors:
    print("\n".join(errors), file=sys.stderr)
    sys.exit(1)
print(f"PASS: {count} archived R04 rows are internally consistent (not GPU revalidation).")
