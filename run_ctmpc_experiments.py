"""Reproducible numerical study for Collective Tube MPC.

This script regenerates the figures and tables used in the extended
IEEE TAC-style manuscript.  It uses only numpy, scipy, pandas, and
matplotlib.  The online MPC problem is implemented as a deterministic
linear-programming tube MPC with tightened state/input constraints.
The LP objective is an l1 tracking objective; the theory in the paper
allows any convex stage cost, and this choice keeps the script portable
(no commercial QP solver or CVXPY is required).  The script also
implements a joint-in-time conformal ellipsoidal confidence-region
baseline in the style of recent conformal SMPC methods.

Run from the package root with

    python code/run_ctmpc_experiments.py

Outputs are written to ./figures and ./results by default.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from shlex import quote as shlex_quote
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import binom


# ----------------------------- system data -----------------------------
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.5], [1.0]])
K = np.array([[-0.593, -1.121]])
AK = A + B @ K
H = 15
X_BOUNDS = np.array([5.0, 2.5])
U_BOUND = 1.2
Q_STAGE = np.diag([1.0, 0.20])
R_STAGE = 0.05

# Disturbance parameters.  The scalar shock creates strong temporal
# dependence along a prediction horizon; the independent vector noise keeps
# the tube genuinely two-dimensional.
DF_T = 4.0
SIGMA_SCALAR_P = 0.075
SIGMA_SCALAR_V = 0.045
SIGMA_IDIO_P = 0.020
SIGMA_IDIO_V = 0.024

METHOD_ORDER = ["collective", "joint_cp", "modular", "robust"]
METHOD_LABEL = {
    "collective": "collective tube",
    "joint_cp": "joint CP ellipsoid",
    "modular": "Bonferroni tube",
    "robust": "sample envelope",
}
LINESTYLE = {
    "collective": "-",
    "joint_cp": "-.",
    "modular": "--",
    "robust": ":",
}
MARKER = {
    "collective": "o",
    "joint_cp": "D",
    "modular": "s",
    "robust": "^",
}
DEFAULT_COLLECTIVE_DISCARDS = 100
NUMERICAL_TOL = 1e-8
K_TERMINAL = np.array([[-1.0, -1.5]])
AF_TERMINAL = A + B @ K_TERMINAL


# -------------------------- calibration tools --------------------------
def conformal_quantile(values: np.ndarray, level: float, axis: int = 0) -> np.ndarray:
    """Upper empirical quantile with the split-conformal ceil convention."""
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"level must be in [0,1], got {level}")
    sorted_values = np.sort(values, axis=axis)
    n = values.shape[axis]
    k = int(math.ceil((n + 1) * level))
    k = min(max(k, 1), n)
    return np.take(sorted_values, k - 1, axis=axis)


def discard_order_statistic(scores: np.ndarray, num_discards: int) -> float:
    """Return the (num_discards+1)-st largest score without interpolation.

    This is the exact order statistic used by the beta-binomial certificate: it
    accepts N-num_discards calibration trajectories and discards the largest
    num_discards scores.  For the paper run, num_discards=100 and therefore
    the certified support/compression size is s=101.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = scores.size
    if not 0 <= num_discards < n:
        raise ValueError(f"num_discards must be in {{0,...,{n-1}}}, got {num_discards}")
    return float(np.partition(scores, n - num_discards - 1)[n - num_discards - 1])


def beta_binomial_bound(N: int, s: int, beta: float) -> float:
    """Smallest eps such that sum_{j=0}^{s-1} C(N,j) eps^j(1-eps)^(N-j) <= beta."""
    if s <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if binom.cdf(s - 1, N, mid) <= beta:
            hi = mid
        else:
            lo = mid
    return hi


def generate_error_trajectories(num: int, horizon: int, rho: float, seed: int) -> np.ndarray:
    """Draw finite-horizon error trajectories e_1,...,e_H.

    A normalized Student-t scalar shock is shared across the horizon with
    weight sqrt(rho), which induces positive temporal dependence.  The error
    trajectory is then propagated through e_{t+1}=(A+BK)e_t+w_t.
    """
    rng = np.random.default_rng(seed)
    common = rng.standard_t(DF_T, size=(num, 1)) / math.sqrt(DF_T / (DF_T - 2.0))
    local = rng.standard_t(DF_T, size=(num, horizon)) / math.sqrt(DF_T / (DF_T - 2.0))
    shocks = math.sqrt(rho) * common + math.sqrt(max(0.0, 1.0 - rho)) * local
    eta = rng.standard_normal((num, horizon, 2))

    w = np.zeros((num, horizon, 2))
    w[:, :, 0] = SIGMA_SCALAR_P * shocks + SIGMA_IDIO_P * eta[:, :, 0]
    w[:, :, 1] = SIGMA_SCALAR_V * shocks + SIGMA_IDIO_V * eta[:, :, 1]

    e = np.zeros((num, 2))
    E = np.zeros((num, horizon, 2))
    for t in range(horizon):
        e = (AK @ e.T).T + w[:, t, :]
        E[:, t, :] = e
    return E


def generate_closed_loop_disturbances(T: int, rho: float, seed: int) -> np.ndarray:
    """Draw a closed-loop disturbance sequence with the same marginal law."""
    rng = np.random.default_rng(seed)
    common = rng.standard_t(DF_T) / math.sqrt(DF_T / (DF_T - 2.0))
    local = rng.standard_t(DF_T, size=T) / math.sqrt(DF_T / (DF_T - 2.0))
    shocks = math.sqrt(rho) * common + math.sqrt(max(0.0, 1.0 - rho)) * local
    eta = rng.standard_normal((T, 2))
    w = np.zeros((T, 2))
    w[:, 0] = SIGMA_SCALAR_P * shocks + SIGMA_IDIO_P * eta[:, 0]
    w[:, 1] = SIGMA_SCALAR_V * shocks + SIGMA_IDIO_V * eta[:, 1]
    return w


@dataclass
class TubeSet:
    tubes: Dict[str, np.ndarray]
    score_scale: np.ndarray
    collective_score: float
    collective_level: float
    collective_discards: int
    joint_cp_score: float
    eps: float


def _joint_cp_ellipsoidal_tube(E_tune: np.ndarray, E_cal: np.ndarray, eps: float) -> Tuple[np.ndarray, float]:
    """Joint-in-time conformal ellipsoidal baseline.

    This implements the trajectory-wise Mahalanobis score used by recent
    conformal SMPC methods: means and covariances are estimated on a tuning
    split, the maximum-in-time squared Mahalanobis score is calibrated on an
    independent split, and each ellipsoid is projected onto the coordinate
    axes for the deterministic box tightening used by this script.
    """
    horizon = E_tune.shape[1]
    mu = np.mean(E_tune, axis=0)
    covs = np.zeros((horizon, 2, 2))
    inv_covs = np.zeros_like(covs)
    for t in range(horizon):
        cov = np.cov(E_tune[:, t, :].T, bias=False)
        # Small ridge for numerical stability.  The ridge is negligible relative
        # to the empirical variances but makes the baseline fully reproducible.
        cov = cov + 1e-10 * np.eye(2)
        covs[t] = cov
        inv_covs[t] = np.linalg.pinv(cov)

    scores = np.zeros(E_cal.shape[0])
    for i, traj in enumerate(E_cal):
        vals = []
        for t in range(horizon):
            d = traj[t] - mu[t]
            vals.append(float(d @ inv_covs[t] @ d))
        scores[i] = max(vals)
    q = float(conformal_quantile(scores, 1.0 - eps, axis=0))

    # Projection of {e : (e-mu)' Sigma^{-1} (e-mu) <= q} onto coordinates.
    # For a centered ellipsoid this is sqrt(q * Sigma_jj); the absolute mean
    # offset is added because the tightened constraints are symmetric boxes.
    widths = np.abs(mu) + np.sqrt(np.maximum(q * np.stack([covs[:, 0, 0], covs[:, 1, 1]], axis=1), 0.0))
    return widths, q


def calibrate_tubes(
    E_tune: np.ndarray,
    E_cal: np.ndarray,
    eps: float = 0.05,
    collective_level: float = 0.96,
    collective_discards: int = DEFAULT_COLLECTIVE_DISCARDS,
) -> TubeSet:
    """Calibrate collective, joint-CP, modular, and sample-envelope tubes.

    E_tune is used only for choosing score geometry.  E_cal is the certification
    split.  Keeping these roles separate makes the finite-sample statements in
    the paper reproducible from the code.
    """
    N, horizon, state_dim = E_cal.shape
    if state_dim != 2:
        raise ValueError("This script assumes the double-integrator state dimension n=2.")
    num_blocks = horizon * state_dim  # box component events |e_{j,t}| <= b_{j,t}

    modular = conformal_quantile(np.abs(E_cal), 1.0 - eps / num_blocks, axis=0)
    score_scale = conformal_quantile(np.abs(E_tune), 0.75, axis=0) + 1e-12
    scores = np.max(np.abs(E_cal) / score_scale[None, :, :], axis=(1, 2))
    # Exact discard-order statistic: the (r+1)-st largest score, not an
    # interpolated numerical quantile.  With N=2500 and r=100, this accepts
    # 2400 calibration trajectories and gives s=r+1=101.
    q = discard_order_statistic(scores, collective_discards)
    collective = q * score_scale
    joint_cp, q_joint = _joint_cp_ellipsoidal_tube(E_tune, E_cal, eps)
    robust = np.max(np.abs(E_cal), axis=0)
    return TubeSet(
        tubes={"collective": collective, "joint_cp": joint_cp, "modular": modular, "robust": robust},
        score_scale=score_scale,
        collective_score=q,
        collective_level=collective_level,
        collective_discards=collective_discards,
        joint_cp_score=q_joint,
        eps=eps,
    )


def tube_violation_risk(E_test: np.ndarray, tube: np.ndarray) -> float:
    return float(np.mean(np.any(np.abs(E_test) > tube[None, :, :] + 1e-12, axis=(1, 2))))


def tube_sum(tube: np.ndarray) -> float:
    return float(np.sum(tube))


def tube_area(tube: np.ndarray) -> float:
    # Area of the rectangular state-error cross-section [-b_p,b_p]x[-b_v,b_v].
    return float(np.mean(4.0 * tube[:, 0] * tube[:, 1]))


def min_input_slack(tube: np.ndarray) -> float:
    margins = [0.0]
    for t in range(1, H):
        b = tube[t - 1, :]
        margins.append(abs(K[0, 0]) * b[0] + abs(K[0, 1]) * b[1])
    return float(U_BOUND - np.max(margins))


# --------------------- admissibility and terminal checks ---------------------
def support_shift_residuals_box(tube: np.ndarray) -> np.ndarray:
    """Residuals for box support-shift inclusions.

    For S_t=[-b_t,b_t], Definition 6 reduces componentwise to

        |A_K^j| b_1 + b_j <= b_{j+1},  j=0,...,H-1,

    with b_0=0.  The returned array is lhs-rhs for each j and coordinate; a
    tube is shift-compatible when all entries are <= numerical tolerance.
    """
    tube = np.asarray(tube, dtype=float)
    if tube.shape != (H, 2):
        raise ValueError(f"expected tube shape {(H, 2)}, got {tube.shape}")
    sections = [np.zeros(2)] + [tube[t] for t in range(H)]
    residuals = []
    for j in range(H):
        lhs = np.abs(np.linalg.matrix_power(AK, j)) @ sections[1] + sections[j]
        rhs = sections[j + 1]
        residuals.append(lhs - rhs)
    return np.asarray(residuals)


def shift_compatible_box_closure(tube: np.ndarray) -> np.ndarray:
    """Smallest componentwise support-shift closure of an axis-aligned box tube.

    The closure contains the raw box tube, keeps S_1 fixed, and recursively
    enlarges later half-widths until all Definition-6 support-shift inequalities
    hold.  The recursion is componentwise minimal under the fixed S_1.
    """
    closed = np.asarray(tube, dtype=float).copy()
    if closed.shape != (H, 2):
        raise ValueError(f"expected tube shape {(H, 2)}, got {closed.shape}")
    b1 = closed[0].copy()
    for j in range(1, H):
        required = closed[j - 1] + np.abs(np.linalg.matrix_power(AK, j)) @ b1
        closed[j] = np.maximum(closed[j], required)
    return closed


def support_D_terminal(direction: np.ndarray, b1: np.ndarray) -> float:
    """Support of D=A_K^(H-1) S_1 in a given direction."""
    direction = np.asarray(direction, dtype=float).reshape(2)
    reach = np.linalg.matrix_power(AK, H - 1)
    return float(np.abs(direction @ reach) @ b1)


def terminal_polytope_constraints(tube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return F,g for the tube-dependent terminal polytope F z <= g.

    The terminal gain is K_TERMINAL=[-1,-1.5], for which AF_TERMINAL^2=0.  Let
    D=A_K^(H-1)S_1.  We first form the set C of terminal states satisfying the
    terminal safety, shifted-state, and shifted-input inequalities in
    Assumption 7.  The terminal set used by the LP is

        X_f = C intersection {z : AF_TERMINAL (z + D) subset C}.

    This makes the Assumption-7 inclusions checkable by linear programs.
    """
    tube = np.asarray(tube, dtype=float)
    if tube.shape != (H, 2):
        raise ValueError(f"expected tube shape {(H, 2)}, got {tube.shape}")
    b1 = tube[0]
    b_hm1 = tube[H - 2]
    b_h = tube[H - 1]

    reach_h = np.abs(np.linalg.matrix_power(AK, H - 1)) @ b1
    state_bound = np.minimum(X_BOUNDS - b_h, X_BOUNDS - b_hm1 - reach_h)

    rows: List[np.ndarray] = []
    rhs: List[float] = []
    for coord in range(2):
        unit = np.zeros(2)
        unit[coord] = 1.0
        rows.extend([unit, -unit])
        rhs.extend([float(state_bound[coord]), float(state_bound[coord])])

    input_margin = (
        U_BOUND
        - float(np.abs(K).reshape(-1) @ b_hm1)
        - support_D_terminal(K_TERMINAL.reshape(-1), b1)
    )
    rows.extend([K_TERMINAL.reshape(-1), -K_TERMINAL.reshape(-1)])
    rhs.extend([float(input_margin), float(input_margin)])

    F_c = np.vstack(rows)
    g_c = np.asarray(rhs)

    pre_rows: List[np.ndarray] = []
    pre_rhs: List[float] = []
    for f_i, g_i in zip(F_c, g_c):
        pre_rows.append(f_i @ AF_TERMINAL)
        pre_rhs.append(float(g_i - support_D_terminal(f_i @ AF_TERMINAL, b1)))

    F = np.vstack([F_c, np.vstack(pre_rows)])
    g = np.concatenate([g_c, np.asarray(pre_rhs)])
    return F, g


def _max_linear_over_polytope(F: np.ndarray, g: np.ndarray, c: np.ndarray) -> float:
    """Maximize c'z over Fz<=g; used only for deterministic checks."""
    res = linprog(-np.asarray(c, dtype=float), A_ub=F, b_ub=g, bounds=[(None, None), (None, None)], method="highs")
    if not res.success:
        return float("nan")
    return float(-res.fun)


def terminal_inclusion_residuals(tube: np.ndarray) -> Dict[str, float]:
    """Numerically verify the terminal inclusions for a box tube.

    Returns maximum lhs-rhs residuals.  Values <= NUMERICAL_TOL mean that the
    corresponding inclusion is verified to numerical precision.
    """
    F, g = terminal_polytope_constraints(tube)
    b1 = tube[0]
    b_hm1 = tube[H - 2]
    b_h = tube[H - 1]

    residual: Dict[str, float] = {}
    horizon_vals = []
    shifted_state_vals = []
    for coord in range(2):
        for sign in (1.0, -1.0):
            direction = np.zeros(2)
            direction[coord] = sign
            max_state = _max_linear_over_polytope(F, g, direction)
            horizon_vals.append(max_state + b_h[coord] - X_BOUNDS[coord])
            shifted_state_vals.append(
                max_state + support_D_terminal(direction, b1) + b_hm1[coord] - X_BOUNDS[coord]
            )
    residual["terminal_horizon_state"] = float(np.nanmax(horizon_vals))
    residual["terminal_shifted_state"] = float(np.nanmax(shifted_state_vals))

    input_vals = []
    input_tightening = float(np.abs(K).reshape(-1) @ b_hm1)
    terminal_shift = support_D_terminal(K_TERMINAL.reshape(-1), b1)
    for sign in (1.0, -1.0):
        direction = sign * K_TERMINAL.reshape(-1)
        input_vals.append(_max_linear_over_polytope(F, g, direction) + terminal_shift + input_tightening - U_BOUND)
    residual["terminal_input"] = float(np.nanmax(input_vals))

    inv_vals = []
    for f_i, g_i in zip(F, g):
        inv_vals.append(_max_linear_over_polytope(F, g, f_i @ AF_TERMINAL) + support_D_terminal(f_i @ AF_TERMINAL, b1) - g_i)
    residual["terminal_invariance"] = float(np.nanmax(inv_vals))
    residual["terminal_polytope_min_rhs"] = float(np.min(g))
    return residual


# ----------------------------- MPC solver -----------------------------
def prediction_matrices(A_: np.ndarray, B_: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    n = A_.shape[0]
    m = B_.shape[1]
    Sx = np.zeros(((horizon + 1) * n, n))
    Su = np.zeros(((horizon + 1) * n, horizon * m))
    Sx[:n, :] = np.eye(n)
    for t in range(1, horizon + 1):
        Sx[t * n : (t + 1) * n, :] = np.linalg.matrix_power(A_, t)
        for i in range(t):
            Su[t * n : (t + 1) * n, i * m : (i + 1) * m] = np.linalg.matrix_power(A_, t - 1 - i) @ B_
    return Sx, Su


class TubeMpcLp:
    """Small deterministic tube-MPC LP used for the closed-loop tests."""

    def __init__(self, horizon: int = H, terminal_bounds: Iterable[float] | None = None) -> None:
        self.horizon = horizon
        self.Sx, self.Su = prediction_matrices(A, B, horizon)
        # terminal_bounds is kept only for backward-compatible construction; the
        # solver below uses the tube-dependent polytope returned by
        # terminal_polytope_constraints, which is the set verified against
        # Assumption 7 in the paper.
        self.terminal_bounds = None if terminal_bounds is None else np.asarray(list(terminal_bounds), dtype=float)
        self.nv = horizon
        self.nxabs = 2 * horizon
        self.nuabs = horizon
        self.nvar = self.nv + self.nxabs + self.nuabs
        self.xabs_start = self.nv
        self.uabs_start = self.nv + self.nxabs

        self.c = np.zeros(self.nvar)
        for t in range(1, horizon + 1):
            weights = np.array([1.0, 0.25])
            if t == horizon:
                weights = np.array([8.0, 2.0])
            self.c[self.xabs_start + (t - 1) * 2 : self.xabs_start + t * 2] = weights
        self.c[self.uabs_start : self.uabs_start + horizon] = 0.05

    def solve(self, x0: np.ndarray, tube: np.ndarray, terminal: bool = True) -> Tuple[np.ndarray | None, str, float]:
        horizon = self.horizon
        input_margin = np.zeros(horizon)
        for t in range(1, horizon):
            b = tube[t - 1, :]
            input_margin[t] = abs(K[0, 0]) * b[0] + abs(K[0, 1]) * b[1]
        u_tight = U_BOUND - input_margin
        if np.any(u_tight <= 1e-9):
            return None, "input tightening empty", np.nan

        bounds = [(-u_tight[t], u_tight[t]) for t in range(horizon)]
        bounds.extend([(0.0, None) for _ in range(self.nxabs + self.nuabs)])

        A_ub: List[np.ndarray] = []
        b_ub: List[float] = []
        v_slice = slice(0, self.nv)

        for t in range(1, horizon + 1):
            state_tight = X_BOUNDS - tube[t - 1, :]
            if np.any(state_tight <= 1e-9):
                return None, "state tightening empty", np.nan

            M = self.Su[t * 2 : (t + 1) * 2, :]
            c0 = self.Sx[t * 2 : (t + 1) * 2, :] @ x0
            for j in range(2):
                row = np.zeros(self.nvar)
                row[v_slice] = M[j]
                A_ub.append(row)
                b_ub.append(float(state_tight[j] - c0[j]))

                row = np.zeros(self.nvar)
                row[v_slice] = -M[j]
                A_ub.append(row)
                b_ub.append(float(state_tight[j] + c0[j]))

                xi = self.xabs_start + (t - 1) * 2 + j
                row = np.zeros(self.nvar)
                row[v_slice] = M[j]
                row[xi] = -1.0
                A_ub.append(row)
                b_ub.append(float(-c0[j]))

                row = np.zeros(self.nvar)
                row[v_slice] = -M[j]
                row[xi] = -1.0
                A_ub.append(row)
                b_ub.append(float(c0[j]))

        if terminal:
            F_term, g_term = terminal_polytope_constraints(tube)
            M = self.Su[horizon * 2 : (horizon + 1) * 2, :]
            c0 = self.Sx[horizon * 2 : (horizon + 1) * 2, :] @ x0
            for f_i, g_i in zip(F_term, g_term):
                row = np.zeros(self.nvar)
                row[v_slice] = f_i @ M
                A_ub.append(row)
                b_ub.append(float(g_i - f_i @ c0))

        for t in range(horizon):
            eta = self.uabs_start + t
            row = np.zeros(self.nvar)
            row[t] = 1.0
            row[eta] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

            row = np.zeros(self.nvar)
            row[t] = -1.0
            row[eta] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

        result = linprog(
            self.c,
            A_ub=np.vstack(A_ub),
            b_ub=np.asarray(b_ub),
            bounds=bounds,
            method="highs",
            options={"time_limit": 5.0},
        )
        if not result.success:
            return None, result.message, np.nan
        return np.asarray(result.x[: self.nv]), "ok", float(result.fun)


def rollout_closed_loop(
    solver: TubeMpcLp,
    tube: np.ndarray,
    rho: float,
    x_initial: np.ndarray,
    T: int,
    num_rollouts: int,
    seed0: int,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    rows = []
    saved_x: List[np.ndarray] = []
    saved_u: List[np.ndarray] = []
    for i in range(num_rollouts):
        x = np.array(x_initial, dtype=float)
        w = generate_closed_loop_disturbances(T, rho, seed0 + i)
        xs = [x.copy()]
        us: List[float] = []
        values: List[float] = []
        cost = 0.0
        state_violation = False
        input_violation = False
        recursive_failure = False

        for k in range(T):
            if np.any(np.abs(x) > X_BOUNDS + 1e-9):
                state_violation = True
                recursive_failure = True
                break

            v, status, value = solver.solve(x, tube, terminal=True)
            if v is None:
                recursive_failure = True
                break

            u = float(v[0])
            us.append(u)
            values.append(value)
            input_violation = input_violation or (abs(u) > U_BOUND + 1e-9)
            cost += float(x @ Q_STAGE @ x + R_STAGE * u * u)
            x = A @ x + B[:, 0] * u + w[k, :]
            xs.append(x.copy())

        if np.any(np.abs(x) > X_BOUNDS + 1e-9):
            state_violation = True
        rows.append(
            {
                "rollout": i,
                "closed_loop_cost": cost,
                "recursive_failure": int(recursive_failure),
                "state_violation": int(state_violation),
                "input_violation": int(input_violation),
                "steps_completed": len(us),
                "peak_abs_input": float(np.max(np.abs(us))) if us else np.nan,
                "mean_lp_value": float(np.mean(values)) if values else np.nan,
            }
        )
        if len(saved_x) < 16:
            saved_x.append(np.asarray(xs))
            saved_u.append(np.asarray(us))
    return pd.DataFrame(rows), saved_x, saved_u


def feasibility_grid(solver: TubeMpcLp, tube: np.ndarray, p_grid: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
    feasible = np.zeros((len(v_grid), len(p_grid)), dtype=bool)
    for i_v, vel in enumerate(v_grid):
        for i_p, pos in enumerate(p_grid):
            sol, status, _ = solver.solve(np.array([pos, vel]), tube, terminal=True)
            feasible[i_v, i_p] = sol is not None
    return feasible


def _write_trajectories_npz(path: Path, trajectories: List[np.ndarray]) -> None:
    payload = {f"x_{idx:02d}": arr for idx, arr in enumerate(trajectories)}
    np.savez(path, **payload)


def _read_trajectories_npz(path: Path) -> List[np.ndarray]:
    if not path.exists():
        return []
    data = np.load(path)
    return [data[key] for key in sorted(data.files)]


def run_closed_loop_worker_cli(
    method: str,
    tube_file: Path,
    worker_dir: Path,
    rho: float,
    steps: int,
    rollouts: int,
    seed0: int,
) -> None:
    """CLI worker for a single closed-loop method."""
    worker_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(tube_file)
    tube = data["tube"]
    x_initial = data["x_initial"]
    solver = TubeMpcLp(H)
    df, traj_x, traj_u = rollout_closed_loop(
        solver,
        tube,
        rho=rho,
        x_initial=x_initial,
        T=steps,
        num_rollouts=rollouts,
        seed0=seed0,
    )
    df.insert(0, "method", method)
    df.to_csv(worker_dir / f"closed_loop_{method}.csv", index=False)
    _write_trajectories_npz(worker_dir / f"traj_x_{method}.npz", traj_x)


def _run_closed_loop_parallel_subprocess(
    tasks: List[Tuple[str, np.ndarray, float, np.ndarray, int, int, int]],
    worker_dir: Path,
) -> List[Tuple[str, pd.DataFrame, List[np.ndarray]]]:
    """Run closed-loop methods concurrently in fresh interpreters.

    A small shell parent launches all workers before waiting.  This is more
    robust than sequential calls on platforms where many HiGHS solves or repeated
    Python subprocess launches can stall.
    """
    worker_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for method, tube, rho, x_initial, steps, runs, seed0 in tasks:
        tube_file = worker_dir / f"tube_closed_loop_{method}.npz"
        np.savez(tube_file, tube=tube, x_initial=x_initial)
        cmd = [
            sys.executable,
            str(script),
            "--worker-mode",
            "closed-loop",
            "--method-name",
            method,
            "--tube-file",
            str(tube_file),
            "--worker-dir",
            str(worker_dir),
            "--rho",
            str(rho),
            "--steps",
            str(steps),
            "--rollouts",
            str(runs),
            "--seed0",
            str(seed0),
        ]
        log_file = worker_dir / f"closed_loop_{method}.log"
        lines.append(" ".join(shlex_quote(part) for part in cmd) + f" > {shlex_quote(str(log_file))} 2>&1 &")
    lines.append("wait")
    shell_file = worker_dir / "run_closed_loop_workers_parallel.sh"
    shell_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shell_file.chmod(0o755)
    subprocess.run(["bash", str(shell_file)], check=True)

    results = []
    for method, *_ in tasks:
        df = pd.read_csv(worker_dir / f"closed_loop_{method}.csv")
        traj_x = _read_trajectories_npz(worker_dir / f"traj_x_{method}.npz")
        results.append((method, df, traj_x))
    return results




def run_grid_worker_cli(method: str, tube_file: Path, worker_dir: Path) -> None:
    """CLI worker for one feasible-grid method in a fresh interpreter."""
    worker_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(tube_file)
    tube = data["tube"]
    p_grid = data["p_grid"]
    v_grid = data["v_grid"]
    solver = TubeMpcLp(H)
    feasible = feasibility_grid(solver, tube, p_grid, v_grid)
    np.savez(worker_dir / f"grid_{method}.npz", feasible=feasible)


def _run_grid_subprocess(method: str, tube: np.ndarray, p_grid: np.ndarray, v_grid: np.ndarray, worker_dir: Path) -> Tuple[str, np.ndarray]:
    """Run one feasible-grid method in a fresh Python interpreter.

    This avoids rare long HiGHS stalls after thousands of LP solves in the main
    process during the preceding closed-loop simulations.
    """
    worker_dir.mkdir(parents=True, exist_ok=True)
    tube_file = worker_dir / f"tube_grid_{method}.npz"
    np.savez(tube_file, tube=tube, p_grid=p_grid, v_grid=v_grid)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        "grid",
        "--method-name",
        method,
        "--tube-file",
        str(tube_file),
        "--worker-dir",
        str(worker_dir),
    ]
    subprocess.run(cmd, check=True)
    data = np.load(worker_dir / f"grid_{method}.npz")
    return method, data["feasible"]


# ----------------------------- figure code -----------------------------
def _get_pyplot():
    """Import matplotlib lazily so worker processes do not touch the font cache."""
    import matplotlib.pyplot as plt  # local import by design
    return plt


def setup_matplotlib() -> None:
    plt = _get_pyplot()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.figsize": (3.5, 2.4),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def plot_tube_profiles(tubes: Dict[str, np.ndarray], out_path: Path) -> None:
    plt = _get_pyplot()
    t = np.arange(1, H + 1)
    fig, axes = plt.subplots(3, 1, figsize=(3.55, 4.3), sharex=True)
    for method in METHOD_ORDER:
        tube = tubes[method]
        axes[0].plot(t, tube[:, 0], LINESTYLE[method], marker=MARKER[method], markersize=2.2, label=METHOD_LABEL[method])
        axes[1].plot(t, tube[:, 1], LINESTYLE[method], marker=MARKER[method], markersize=2.2)
        margins = np.zeros(H)
        for tau in range(1, H):
            b = tube[tau - 1, :]
            margins[tau] = abs(K[0, 0]) * b[0] + abs(K[0, 1]) * b[1]
        axes[2].plot(np.arange(0, H), U_BOUND - margins, LINESTYLE[method], marker=MARKER[method], markersize=2.2)
    axes[0].set_ylabel(r"$b_p(t)$")
    axes[1].set_ylabel(r"$b_v(t)$")
    axes[2].set_ylabel(r"$u$ slack")
    axes[2].set_xlabel("prediction step")
    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    for ax in axes:
        ax.grid(True, linewidth=0.25, alpha=0.5)
    fig.savefig(out_path)
    plt.close(fig)


def plot_dependence_sweep(df: pd.DataFrame, out_path: Path) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45))
    for method in METHOD_ORDER:
        d = df[df["method"] == method].sort_values("rho")
        axes[0].plot(d["rho"], d["empirical_joint_risk"], LINESTYLE[method], marker=MARKER[method], markersize=3, label=METHOD_LABEL[method])
    axes[0].axhline(0.05, color="0.2", linewidth=0.6)
    axes[0].set_xlabel(r"temporal dependence rho")
    axes[0].set_ylabel("joint tube-escape risk")
    axes[0].set_ylim(0, max(0.06, 1.15 * df["empirical_joint_risk"].max()))
    axes[0].grid(True, linewidth=0.25, alpha=0.5)
    axes[0].legend(loc="upper right", frameon=False, fontsize=6)

    d = df[df["method"] == "collective"].sort_values("rho")
    axes[1].plot(d["rho"], 100.0 * d["sum_reduction_vs_modular"], "-o", markersize=3, label="sum reduction")
    axes[1].plot(d["rho"], 100.0 * d["area_reduction_vs_modular"], "--s", markersize=3, label="area reduction")
    axes[1].set_xlabel(r"temporal dependence rho")
    axes[1].set_ylabel("collective reduction [%]")
    axes[1].grid(True, linewidth=0.25, alpha=0.5)
    axes[1].legend(loc="upper left", frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_feasible_region(
    p_grid: np.ndarray,
    v_grid: np.ndarray,
    feasible_collective: np.ndarray,
    feasible_modular: np.ndarray,
    out_path: Path,
) -> None:
    plt = _get_pyplot()
    P, V = np.meshgrid(p_grid, v_grid)
    collective_only = feasible_collective & (~feasible_modular)
    fig, ax = plt.subplots(figsize=(3.55, 2.95))
    ax.scatter(
        P[feasible_modular],
        V[feasible_modular],
        s=10,
        marker="s",
        color="0.45",
        alpha=0.28,
        label=f"Bonferroni feasible ({int(np.sum(feasible_modular))})",
    )
    ax.scatter(
        P[collective_only],
        V[collective_only],
        s=16,
        marker="o",
        color="C1",
        alpha=0.9,
        label=f"collective only ({int(np.sum(collective_only))})",
    )
    ax.plot(
        [-X_BOUNDS[0], X_BOUNDS[0], X_BOUNDS[0], -X_BOUNDS[0], -X_BOUNDS[0]],
        [-X_BOUNDS[1], -X_BOUNDS[1], X_BOUNDS[1], X_BOUNDS[1], -X_BOUNDS[1]],
        linestyle="--",
        color="0.15",
        linewidth=0.8,
        label="hard state box",
    )
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_xlim(-5.45, 5.45)
    ax.set_ylim(-2.85, 2.85)
    ax.grid(True, linewidth=0.25, alpha=0.45)
    ax.legend(loc="lower right", frameon=True, fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_closed_loop_phase(traj: Dict[str, List[np.ndarray]], out_path: Path) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharex=True, sharey=True)
    for ax, method in zip(axes, ["collective", "modular"]):
        for xs in traj[method]:
            ax.plot(xs[:, 0], xs[:, 1], linewidth=0.7, alpha=0.75)
            ax.plot(xs[0, 0], xs[0, 1], marker="o", markersize=2.5)
            ax.plot(xs[-1, 0], xs[-1, 1], marker="x", markersize=2.5)
        ax.plot([-X_BOUNDS[0], X_BOUNDS[0], X_BOUNDS[0], -X_BOUNDS[0], -X_BOUNDS[0]],
                [-X_BOUNDS[1], -X_BOUNDS[1], X_BOUNDS[1], X_BOUNDS[1], -X_BOUNDS[1]],
                linewidth=0.8)
        ax.set_title(METHOD_LABEL[method])
        ax.set_xlabel("position")
        ax.grid(True, linewidth=0.25, alpha=0.5)
    axes[0].set_ylabel("velocity")
    fig.savefig(out_path)
    plt.close(fig)


def plot_closed_loop_bands(summary_by_method: Dict[str, List[np.ndarray]], out_path: Path) -> None:
    plt = _get_pyplot()
    # summary_by_method contains complete state trajectories, padded/truncated to common T+1.
    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.35), sharex=True)
    for method in ["collective", "modular"]:
        X = np.stack([x for x in summary_by_method[method] if len(x) == len(summary_by_method[method][0])], axis=0)
        t = np.arange(X.shape[1])
        median = np.median(X, axis=0)
        lo = np.quantile(X, 0.10, axis=0)
        hi = np.quantile(X, 0.90, axis=0)
        axes[0].plot(t, median[:, 0], LINESTYLE[method], label=METHOD_LABEL[method])
        axes[0].fill_between(t, lo[:, 0], hi[:, 0], alpha=0.12)
        axes[1].plot(t, median[:, 1], LINESTYLE[method])
        axes[1].fill_between(t, lo[:, 1], hi[:, 1], alpha=0.12)
    axes[0].axhline(X_BOUNDS[0], linewidth=0.5)
    axes[0].axhline(-X_BOUNDS[0], linewidth=0.5)
    axes[1].axhline(X_BOUNDS[1], linewidth=0.5)
    axes[1].axhline(-X_BOUNDS[1], linewidth=0.5)
    axes[0].set_ylabel("position")
    axes[1].set_ylabel("velocity")
    axes[1].set_xlabel("closed-loop time")
    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    for ax in axes:
        ax.grid(True, linewidth=0.25, alpha=0.5)
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------- main ---------------------------------
def run_experiments(out_dir: Path, quick: bool = False, calibration_only: bool = False) -> None:
    figures_dir = out_dir / "figures"
    results_dir = out_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    eps = 0.05
    beta = 0.05
    N_tune = 2500 if not quick else 800
    N_cal = 2500 if not quick else 800
    N_test = 80000 if not quick else 20000
    collective_discards = DEFAULT_COLLECTIVE_DISCARDS if not quick else 32
    collective_level = 1.0 - collective_discards / N_cal
    rhos = [0.0, 0.3, 0.6, 0.85, 0.95]

    open_rows = []
    admissibility_rows = []
    raw_tubes_at_085: Dict[str, np.ndarray] | None = None
    mpc_tubes_at_085: Dict[str, np.ndarray] | None = None
    for rho in rhos:
        E_tune = generate_error_trajectories(N_tune, H, rho, seed=50 + int(100 * rho))
        E_cal = generate_error_trajectories(N_cal, H, rho, seed=100 + int(100 * rho))
        E_test = generate_error_trajectories(N_test, H, rho, seed=200 + int(100 * rho))
        tube_set = calibrate_tubes(
            E_tune,
            E_cal,
            eps=eps,
            collective_level=collective_level,
            collective_discards=collective_discards,
        )

        # The raw collective score box is the calibrated statistical object.
        # Before solving the collective MPC problem, it is replaced by the
        # smallest componentwise shift-compatible closure.  Containment is
        # preserved, so the finite-sample risk certificate remains valid.  The
        # other methods remain the raw baselines used for empirical comparison.
        mpc_tubes = dict(tube_set.tubes)
        mpc_tubes["collective"] = shift_compatible_box_closure(tube_set.tubes["collective"])

        method = "collective"
        raw_shift = support_shift_residuals_box(tube_set.tubes[method])
        closed_shift = support_shift_residuals_box(mpc_tubes[method])
        terminal_res = terminal_inclusion_residuals(mpc_tubes[method])
        admissibility_rows.append(
            {
                "rho": rho,
                "method": method,
                "raw_max_shift_residual": float(np.max(raw_shift)),
                "closed_max_shift_residual": float(np.max(closed_shift)),
                "closed_max_added_half_width": float(np.max(mpc_tubes[method] - tube_set.tubes[method])),
                "raw_sum_half_widths": tube_sum(tube_set.tubes[method]),
                "closed_sum_half_widths": tube_sum(mpc_tubes[method]),
                "raw_min_input_slack": min_input_slack(tube_set.tubes[method]),
                "closed_min_input_slack": min_input_slack(mpc_tubes[method]),
                **terminal_res,
            }
        )

        if abs(rho - 0.85) < 1e-12:
            raw_tubes_at_085 = tube_set.tubes
            mpc_tubes_at_085 = mpc_tubes

        modular_sum = tube_sum(tube_set.tubes["modular"])
        modular_area = tube_area(tube_set.tubes["modular"])
        for method in METHOD_ORDER:
            tube = tube_set.tubes[method]
            open_rows.append(
                {
                    "rho": rho,
                    "method": method,
                    "empirical_joint_risk": tube_violation_risk(E_test, tube),
                    "sum_half_widths": tube_sum(tube),
                    "mean_cross_section_area": tube_area(tube),
                    "min_input_slack": min_input_slack(tube),
                    "sum_reduction_vs_modular": 1.0 - tube_sum(tube) / modular_sum,
                    "area_reduction_vs_modular": 1.0 - tube_area(tube) / modular_area,
                }
            )

    open_df = pd.DataFrame(open_rows)
    open_df.to_csv(results_dir / "open_loop_risk_sweep.csv", index=False)
    pd.DataFrame(admissibility_rows).to_csv(results_dir / "admissibility_report.csv", index=False)
    plot_dependence_sweep(open_df, figures_dir / "dependence_sweep.pdf")

    if raw_tubes_at_085 is None or mpc_tubes_at_085 is None:
        raise RuntimeError("rho=0.85 tube was not generated")
    plot_tube_profiles(raw_tubes_at_085, figures_dir / "tube_profiles_rho085.pdf")

    # Certificate table for the beta-binomial risk expression in the paper.
    cert_rows = []
    for s in [1, 2, 5, 10, 25, 50, 101, 126]:
        cert_rows.append({"N": N_cal, "support_or_compression_size": s, "beta": beta, "risk_bound": beta_binomial_bound(N_cal, s, beta)})
    pd.DataFrame(cert_rows).to_csv(results_dir / "certificate_table.csv", index=False)

    # Save the rho=0.85 tubes so grid and closed-loop stages can run in fresh
    # interpreters without repeating calibration.  This keeps the full
    # reproduction robust on platforms where many sequential HiGHS solves in a
    # single process can stall.
    np.savez(
        results_dir / "tubes_rho085.npz",
        **{f"raw_{method}": raw_tubes_at_085[method] for method in METHOD_ORDER},
        **{f"mpc_{method}": mpc_tubes_at_085[method] for method in METHOD_ORDER},
    )
    if calibration_only:
        print("Calibration, figures, certificate, and admissibility report written.")
        print(f"Figures written to {figures_dir}")
        print(f"Tables written to {results_dir}")
        return

    # Feasible-set map.
    p_grid = np.linspace(-5.0, 5.0, 25 if not quick else 15)
    v_grid = np.linspace(-2.5, 2.5, 19 if not quick else 11)
    grid_methods = ["collective", "modular"]
    if quick:
        grid_solver = TubeMpcLp(H)
        feas = {method: feasibility_grid(grid_solver, mpc_tubes_at_085[method], p_grid, v_grid) for method in grid_methods}
    else:
        worker_dir = results_dir / "_worker_outputs"
        feas = {method: grid for method, grid in (_run_grid_subprocess(method, mpc_tubes_at_085[method], p_grid, v_grid, worker_dir) for method in grid_methods)}
    modular_only = feas["modular"] & (~feas["collective"])
    if np.any(modular_only):
        print(f"Warning: {int(np.sum(modular_only))} grid points are modular-feasible but not collective-feasible.")
    np.savez(
        results_dir / "feasibility_grid.npz",
        p_grid=p_grid,
        v_grid=v_grid,
        modular_only_count=int(np.sum(modular_only)),
        **{f"feasible_{m}": feas[m] for m in grid_methods},
    )
    feasible_total = len(p_grid) * len(v_grid)
    feasible_summary = pd.DataFrame(
        [
            {
                "method": method,
                "feasible_points": int(np.sum(feas[method])),
                "grid_points": int(feasible_total),
                "feasible_fraction": float(np.mean(feas[method])),
                "collective_only_points": int(np.sum(feas["collective"] & (~feas["modular"]))) if method == "collective" else 0,
                "modular_only_points": int(np.sum(feas["modular"] & (~feas["collective"]))) if method == "modular" else 0,
            }
            for method in grid_methods
        ]
    )
    feasible_summary.to_csv(results_dir / "feasible_region_summary.csv", index=False)
    plot_feasible_region(p_grid, v_grid, feas["collective"], feas["modular"], figures_dir / "feasible_region_rho085.pdf")


    # Closed-loop MPC tests at the strongest dependence value used for detailed plots.
    solver = TubeMpcLp(H)
    rho_cl = 0.85
    T = 35
    x_initial = np.array([4.8, 0.3])
    num_rollouts = 40 if not quick else 5
    closed_rows = []
    saved_traj: Dict[str, List[np.ndarray]] = {}
    saved_all_traj: Dict[str, List[np.ndarray]] = {}
    for method in METHOD_ORDER:
        this_num = num_rollouts if method != "robust" else (20 if not quick else num_rollouts)
        df, traj_x, traj_u = rollout_closed_loop(
            solver,
            mpc_tubes_at_085[method],
            rho=rho_cl,
            x_initial=x_initial,
            T=T,
            num_rollouts=this_num,
            seed0=9000,
        )
        df.insert(0, "method", method)
        closed_rows.append(df)
        saved_traj[method] = traj_x
        saved_all_traj[method] = traj_x

    closed_df = pd.concat(closed_rows, ignore_index=True)
    closed_df.to_csv(results_dir / "closed_loop_rollouts.csv", index=False)
    closed_summary = closed_df.groupby("method").agg(
        n_rollouts=("rollout", "count"),
        mean_cost=("closed_loop_cost", "mean"),
        std_cost=("closed_loop_cost", "std"),
        recursive_failure_rate=("recursive_failure", "mean"),
        state_violation_rate=("state_violation", "mean"),
        input_violation_rate=("input_violation", "mean"),
        mean_steps_completed=("steps_completed", "mean"),
        peak_abs_input=("peak_abs_input", "mean"),
    ).reset_index()
    # Relative columns against collective for readability.
    coll_cost = float(closed_summary.loc[closed_summary["method"] == "collective", "mean_cost"].iloc[0])
    closed_summary["cost_increase_vs_collective_pct"] = 100.0 * (closed_summary["mean_cost"] / coll_cost - 1.0)
    closed_summary.to_csv(results_dir / "closed_loop_summary.csv", index=False)
    plot_closed_loop_phase(saved_traj, figures_dir / "closed_loop_phase.pdf")

    # Minimal machine-readable reproducibility manifest.
    manifest = pd.DataFrame([
        {"quantity": "N_tune", "value": N_tune},
        {"quantity": "N_cal", "value": N_cal},
        {"quantity": "N_test", "value": N_test},
        {"quantity": "eps", "value": eps},
        {"quantity": "collective_level", "value": collective_level},
        {"quantity": "collective_discards", "value": collective_discards},
        {"quantity": "collective_support_size", "value": collective_discards + 1},
        {"quantity": "horizon", "value": H},
        {"quantity": "closed_loop_rollouts", "value": num_rollouts},
        {"quantity": "closed_loop_steps", "value": T},
    ])
    manifest.to_csv(results_dir / "linear_reproducibility_manifest.csv", index=False)

    # Print compact summaries for command-line reproducibility.
    print("Open-loop sweep:")
    print(open_df.pivot(index="rho", columns="method", values="empirical_joint_risk"))
    print("\nClosed-loop summary:")
    print(closed_summary)
    print("\nFeasible-region summary:")
    print(feasible_summary)
    print(f"\nFigures written to {figures_dir}")
    print(f"Tables written to {results_dir}")


def _load_mpc_tubes(results_dir: Path) -> Dict[str, np.ndarray]:
    tube_file = results_dir / "tubes_rho085.npz"
    if not tube_file.exists():
        raise FileNotFoundError(
            f"{tube_file} not found. Run the calibration stage first, e.g. "
            "python code/run_ctmpc_experiments.py --stage calibration"
        )
    data = np.load(tube_file)
    return {method: data[f"mpc_{method}"] for method in METHOD_ORDER}


def run_grid_stage(out_dir: Path, quick: bool = False) -> None:
    figures_dir = out_dir / "figures"
    results_dir = out_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    mpc_tubes_at_085 = _load_mpc_tubes(results_dir)
    p_grid = np.linspace(-5.0, 5.0, 25 if not quick else 15)
    v_grid = np.linspace(-2.5, 2.5, 19 if not quick else 11)
    grid_methods = ["collective", "modular"]
    solver = TubeMpcLp(H)
    feas = {method: feasibility_grid(solver, mpc_tubes_at_085[method], p_grid, v_grid) for method in grid_methods}
    modular_only = feas["modular"] & (~feas["collective"])
    if np.any(modular_only):
        print(f"Warning: {int(np.sum(modular_only))} grid points are modular-feasible but not collective-feasible.")
    np.savez(
        results_dir / "feasibility_grid.npz",
        p_grid=p_grid,
        v_grid=v_grid,
        modular_only_count=int(np.sum(modular_only)),
        **{f"feasible_{m}": feas[m] for m in grid_methods},
    )
    feasible_total = len(p_grid) * len(v_grid)
    feasible_summary = pd.DataFrame(
        [
            {
                "method": method,
                "feasible_points": int(np.sum(feas[method])),
                "grid_points": int(feasible_total),
                "feasible_fraction": float(np.mean(feas[method])),
                "collective_only_points": int(np.sum(feas["collective"] & (~feas["modular"]))) if method == "collective" else 0,
                "modular_only_points": int(np.sum(feas["modular"] & (~feas["collective"]))) if method == "modular" else 0,
            }
            for method in grid_methods
        ]
    )
    feasible_summary.to_csv(results_dir / "feasible_region_summary.csv", index=False)
    plot_feasible_region(p_grid, v_grid, feas["collective"], feas["modular"], figures_dir / "feasible_region_rho085.pdf")
    print("Feasible-region summary:")
    print(feasible_summary)


def run_closed_loop_stage(out_dir: Path, quick: bool = False) -> None:
    figures_dir = out_dir / "figures"
    results_dir = out_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    mpc_tubes_at_085 = _load_mpc_tubes(results_dir)
    solver = TubeMpcLp(H)
    rho_cl = 0.85
    T = 35
    x_initial = np.array([4.8, 0.3])
    num_rollouts = 40 if not quick else 5
    closed_rows = []
    saved_traj: Dict[str, List[np.ndarray]] = {}
    tasks = []
    for method in METHOD_ORDER:
        this_num = num_rollouts if method != "robust" else (20 if not quick else num_rollouts)
        tasks.append((method, mpc_tubes_at_085[method], rho_cl, x_initial, T, this_num, 9000))
    if quick:
        for method, tube, rho, x0, steps, runs, seed0 in tasks:
            df, traj_x, traj_u = rollout_closed_loop(
                solver,
                tube,
                rho=rho,
                x_initial=x0,
                T=steps,
                num_rollouts=runs,
                seed0=seed0,
            )
            df.insert(0, "method", method)
            closed_rows.append(df)
            saved_traj[method] = traj_x
    else:
        worker_dir = results_dir / "_worker_outputs"
        for method, df, traj_x in _run_closed_loop_parallel_subprocess(tasks, worker_dir):
            closed_rows.append(df)
            saved_traj[method] = traj_x

    closed_df = pd.concat(closed_rows, ignore_index=True)
    closed_df.to_csv(results_dir / "closed_loop_rollouts.csv", index=False)
    closed_summary = closed_df.groupby("method").agg(
        n_rollouts=("rollout", "count"),
        mean_cost=("closed_loop_cost", "mean"),
        std_cost=("closed_loop_cost", "std"),
        recursive_failure_rate=("recursive_failure", "mean"),
        state_violation_rate=("state_violation", "mean"),
        input_violation_rate=("input_violation", "mean"),
        mean_steps_completed=("steps_completed", "mean"),
        peak_abs_input=("peak_abs_input", "mean"),
    ).reset_index()
    coll_cost = float(closed_summary.loc[closed_summary["method"] == "collective", "mean_cost"].iloc[0])
    closed_summary["cost_increase_vs_collective_pct"] = 100.0 * (closed_summary["mean_cost"] / coll_cost - 1.0)
    closed_summary.to_csv(results_dir / "closed_loop_summary.csv", index=False)
    plot_closed_loop_phase(saved_traj, figures_dir / "closed_loop_phase.pdf")
    print("Closed-loop summary:")
    print(closed_summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1], help="Package root/output directory")
    parser.add_argument("--quick", action="store_true", help="Run a shorter version for smoke testing")
    parser.add_argument("--worker-mode", choices=["grid", "closed-loop"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--method-name", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--tube-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rho", type=float, default=0.85, help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=35, help=argparse.SUPPRESS)
    parser.add_argument("--rollouts", type=int, default=40, help=argparse.SUPPRESS)
    parser.add_argument("--seed0", type=int, default=9000, help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=["all", "calibration", "grid", "closed-loop"], default="all", help="run one reproducibility stage")
    args = parser.parse_args()
    if args.worker_mode is not None:
        if args.tube_file is None or args.worker_dir is None or not args.method_name:
            raise ValueError("worker mode requires --method-name, --tube-file, and --worker-dir")
        if args.worker_mode == "grid":
            run_grid_worker_cli(args.method_name, args.tube_file, args.worker_dir)
        elif args.worker_mode == "closed-loop":
            run_closed_loop_worker_cli(
                args.method_name,
                args.tube_file,
                args.worker_dir,
                rho=args.rho,
                steps=args.steps,
                rollouts=args.rollouts,
                seed0=args.seed0,
            )
        return
    if args.stage == "calibration":
        run_experiments(args.out, quick=args.quick, calibration_only=True)
    elif args.stage == "grid":
        run_grid_stage(args.out, quick=args.quick)
    elif args.stage == "closed-loop":
        run_closed_loop_stage(args.out, quick=args.quick)
    else:
        run_experiments(args.out, quick=args.quick)


if __name__ == "__main__":
    main()
