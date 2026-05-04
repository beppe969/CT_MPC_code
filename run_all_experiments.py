"""Run numerical experiments for the CT-MPC TAC pro manuscript package.

Default mode regenerates the figures/results used in the paper by executing the
linear study in three fresh stages: calibration/admissibility, feasible-region
map, and closed-loop rollouts.  The staged execution avoids rare platform-
dependent stalls after many sequential LP solves in one Python interpreter.
Use --quick for a smoke test that writes to ./quick_outputs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="short smoke test written to quick_outputs")
    parser.add_argument(
        "--include-supplemental-nonlinear",
        action="store_true",
        help="also run the supplemental nonlinear drag script from the longer draft",
    )
    parser.add_argument("--chain-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    py = sys.executable
    stages = ["calibration", "grid", "closed-loop"]
    if args.chain_index < len(stages):
        stage = stages[args.chain_index]
        cmd = [py, "code/run_ctmpc_experiments.py", "--stage", stage]
        if args.quick:
            cmd.extend(["--quick", "--out", str(ROOT / "quick_outputs")])
        run(cmd)
        next_index = args.chain_index + 1
        if next_index < len(stages):
            argv = [py, str(Path(__file__).resolve()), "--chain-index", str(next_index)]
            if args.quick:
                argv.append("--quick")
            if args.include_supplemental_nonlinear:
                argv.append("--include-supplemental-nonlinear")
            os.execvpe(py, argv, os.environ)
        # Fall through after the last linear stage to optional supplemental run.

    if args.include_supplemental_nonlinear:
        if args.quick:
            quick_root = ROOT / "quick_outputs"
            run([
                py,
                "code/run_nonlinear_drag_experiment.py",
                "--fig-dir", str(quick_root / "figures"),
                "--results-dir", str(quick_root / "results"),
                "--calib", "400",
                "--test", "1500",
            ])
        else:
            run([py, "code/run_nonlinear_drag_experiment.py"])


if __name__ == "__main__":
    main()
