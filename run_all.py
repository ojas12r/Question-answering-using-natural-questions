#!/usr/bin/env python3
"""
run_all.py — Python entry point that reproduces the full pipeline end-to-end.

Usage:
    python run_all.py [--config config.yaml] [--skip-data] [--skip-index]
                      [--skip-experiments] [--skip-tests]

This is the Python equivalent of `make all`, suitable for Windows where GNU
Make may not be installed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_step(name: str, cmd: list[str], cwd: Path) -> None:
    """Run a pipeline step, print timing, and abort on failure."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f" STEP: {name}")
    print(f"{sep}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{name}' failed (exit code {result.returncode}).")
        sys.exit(result.returncode)
    print(f"\n[OK] '{name}' completed in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full NQ-retrieval pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest")
    parser.add_argument("--skip-data", action="store_true", help="Skip data download/subset")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building")
    parser.add_argument("--skip-experiments", action="store_true", help="Skip ranking & eval")
    args = parser.parse_args()

    root = Path(__file__).parent
    py = sys.executable  # use the same Python that invoked this script

    steps: list[tuple[str, list[str], bool]] = [
        ("Tests (pytest)", [py, "-m", "pytest", "tests/", "-v", "--tb=short"], args.skip_tests),
        ("Dataset build", [py, "src/data.py", "--config", args.config], args.skip_data),
        ("Index build", [py, "src/index.py", "--config", args.config], args.skip_index),
        (
            "Experiments (tune + rank + evaluate)",
            [py, "src/run_experiments.py", "--config", args.config],
            args.skip_experiments,
        ),
    ]

    total_start = time.perf_counter()
    skipped = []
    for name, cmd, skip in steps:
        if skip:
            print(f"[SKIP] {name}")
            skipped.append(name)
            continue
        run_step(name, cmd, root)

    total = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print(f" ALL DONE in {total / 60:.1f} min")
    if skipped:
        print(f" Skipped: {', '.join(skipped)}")
    print(" Results → results/  |  Report support → report/")
    print(" Demo    → streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
