"""Run the numerical experiments for the CT-MPC TAC manuscript package.

Default mode regenerates the figures and CSV tables used in the paper by
executing the linear study in three stages: calibration/admissibility,
feasible-region map, and closed-loop rollouts.  Use --quick for a shorter smoke
test that writes to ./quick_outputs without overwriting the manuscript outputs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import run_ctmpc_experiments as ctmpc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="short smoke test written to quick_outputs")
    args = parser.parse_args()

    out_dir = ROOT / "quick_outputs" if args.quick else ROOT
    print("$ calibration/admissibility stage", flush=True)
    ctmpc.run_experiments(out_dir, quick=args.quick, calibration_only=True)
    print("$ feasible-region stage", flush=True)
    ctmpc.run_grid_stage(out_dir, quick=args.quick)
    print("$ closed-loop stage", flush=True)
    ctmpc.run_closed_loop_stage(out_dir, quick=args.quick)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
