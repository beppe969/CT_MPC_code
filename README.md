# Numerical reproducibility code

Install dependencies from the package root:

```bash
pip install -r CT_MPC_code/requirements.txt
```

Run a quick smoke test without overwriting the paper outputs:

```bash
python CT_MPC_code/run_all_experiments.py --quick
```

Run the full linear study used in the paper:

```bash
python CT_MPC_code/run_all_experiments.py
```

The full driver executes three stages (`calibration`, `grid`, and `closed-loop`).  The same stages can be run one at a time:

```bash
python CT_MPC_code/run_ctmpc_experiments.py --stage calibration
python CT_MPC_code/run_ctmpc_experiments.py --stage grid
python CT_MPC_code/run_ctmpc_experiments.py --stage closed-loop
```

## Main paper script

`run_ctmpc_experiments.py` regenerates the linear double-integrator study. It includes four tubes:

1. Collective tube MPC (CT-MPC), using an independent tuning split for score scales and an independent calibration split for the beta-binomial certificate.
2. Joint-in-time conformal ellipsoidal baseline, implemented from a tuning-split mean/covariance estimate, a trajectory-wise maximum Mahalanobis score, and coordinate projections of the calibrated ellipsoids.
3. Bonferroni component-time tube.
4. Component-wise sample envelope.

The collective threshold is implemented as an exact discard order statistic.  For the paper run, `r=100` and the code uses the `(r+1)`-st largest calibration score, not an interpolated numerical quantile.  The corresponding support/compression size is `s=101`.

The raw collective score box is also checked against the deterministic CT-MPC assumptions.  If it is not shift-compatible, the script uses the smallest componentwise shift-compatible closure for the collective MPC runs.  The file `results/admissibility_report.csv` records the raw support-shift residual, the closed residual, the largest added half-width, and the terminal-inclusion residuals with tolerance `1e-8`.

The linear script writes the paper graphics and tables to `figures/` and `results/`:

- `open_loop_risk_sweep.csv`
- `admissibility_report.csv`
- `certificate_table.csv`
- `closed_loop_summary.csv`
- `closed_loop_rollouts.csv`
- `feasible_region_summary.csv`
- `feasibility_grid.npz`
- `linear_reproducibility_manifest.csv`
- `tubes_rho085.npz`
- `dependence_sweep.pdf`
- `tube_profiles_rho085.pdf`
- `feasible_region_rho085.pdf`

`closed_loop_phase.pdf` is generated as a diagnostic only; it is not included in the manuscript.

## Dependencies

The scripts require NumPy, SciPy, pandas, and Matplotlib only. No commercial solver and no CVXPY installation are required.
