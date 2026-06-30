"""
vln_metrics.py — VLN-CE evaluation metrics, pure NumPy (no ROS dependency).

Reusable offline: feed it a recorded trajectory + the route's reference path
and it returns every metric the benchmark needs. Distances are computed in 2D
(x, y) by default — a ground robot on a flat floor; z is ignored.

References:
  SPL   : Anderson et al. 2018, "On Evaluation of Embodied Navigation Agents".
  nDTW  : Ilharco et al. 2019, "General Evaluation for Instruction Conditioned
          Navigation using Dynamic Time Warping". nDTW = exp(-DTW/(|R|*d_th)).
  SDTW  : success-weighted nDTW = success * nDTW.
"""
from __future__ import annotations
import numpy as np


def _xy(points) -> np.ndarray:
    """Coerce an (N,2) or (N,3) array to (N,2) float; keep only x,y."""
    p = np.asarray(points, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return p[:, :2]


def trajectory_length(traj, min_step: float = 0.003) -> float:
    """
    Total arc length (m) of the *dense* trajectory.

    min_step is a deadband (default 3 mm): segments shorter than this are
    treated as mocap jitter and not accumulated, otherwise sub-millimetre
    noise at 200 Hz integrates into a badly inflated path length.
    """
    p = _xy(traj)
    if len(p) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return float(seg[seg >= min_step].sum())


def navigation_error(traj, goal) -> float:
    """Distance (m) from the FINAL pose to the goal."""
    p = _xy(traj)
    g = _xy(goal).reshape(-1)[:2]
    if len(p) == 0:
        return float("nan")
    return float(np.linalg.norm(p[-1] - g))


def oracle_min_distance(traj, goal) -> float:
    """Closest the robot EVER got to the goal (m), over the whole trajectory."""
    p = _xy(traj)
    g = _xy(goal).reshape(-1)[:2]
    if len(p) == 0:
        return float("nan")
    return float(np.linalg.norm(p - g, axis=1).min())


def resample_by_arclength(traj, spacing: float) -> np.ndarray:
    """
    Resample a trajectory to roughly uniform `spacing` (m) along arc length.
    This brings the predicted path to a resolution comparable to the reference
    path before DTW, so the DTW cost is not dominated by raw sample rate.
    """
    p = _xy(traj)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return p[:1]
    n = max(2, int(np.floor(total / spacing)) + 1)
    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2))
    out[:, 0] = np.interp(targets, s, p[:, 0])
    out[:, 1] = np.interp(targets, s, p[:, 1])
    return out


def dtw_distance(Q, R) -> float:
    """Classic DTW cumulative distance (Euclidean local cost) between Q and R."""
    Q = _xy(Q)
    R = _xy(R)
    n, m = len(Q), len(R)
    if n == 0 or m == 0:
        return float("nan")
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        qi = Q[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(qi - R[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def ndtw(traj, reference, d_th: float, spacing: float | None = 0.25) -> float:
    """
    Normalised DTW in [0, 1]; higher = more faithful to the reference route.
        nDTW = exp( -DTW(Q, R) / (|R| * d_th) )
    `d_th` is the success threshold (same one used for SR).

    Both the trajectory AND the reference are arc-length resampled to `spacing`
    before the DTW. This matters: DTW matches point-to-point, so if the
    reference is given only as sparse landmark corners, every intermediate
    trajectory point sits far from the nearest reference *node* and nDTW
    collapses even for a perfect run. Densifying both to the same spacing
    (as VLN-CE does implicitly, its reference being a dense node sequence)
    makes the score behave: a faithful trajectory -> nDTW ~ 1.
    Pass spacing=None only if your reference is already densely sampled.
    """
    R = _xy(reference)
    if len(R) == 0:
        return float("nan")
    if spacing:
        Q = resample_by_arclength(traj, spacing)
        R = resample_by_arclength(R, spacing)
    else:
        Q = _xy(traj)
    d = dtw_distance(Q, R)
    if not np.isfinite(d):
        return float("nan")
    return float(np.exp(-d / (len(R) * d_th)))


def is_success(traj, goal, d_th: float, stopped: bool) -> int:
    """1 iff the agent stopped on its own (not aborted/timed out) within d_th."""
    if not stopped:
        return 0
    ne = navigation_error(traj, goal)
    return int(np.isfinite(ne) and ne <= d_th)


def oracle_success(traj, goal, d_th: float) -> int:
    """1 iff the robot was ever within d_th of the goal, regardless of stop."""
    omd = oracle_min_distance(traj, goal)
    return int(np.isfinite(omd) and omd <= d_th)


def spl(success: int, ref_len: float, actual_len: float) -> float:
    """Success weighted by Path Length = S * ref / max(actual, ref)."""
    denom = max(actual_len, ref_len)
    if denom <= 0:
        return 0.0
    return float(success * ref_len / denom)


def compute_all(traj, reference, goal, d_th, ref_len=None,
                stopped=True, spacing=0.25):
    """
    One-shot: returns a dict with every per-episode metric.
    `ref_len` defaults to the arc length of the reference polyline if not given.
    """
    if ref_len is None:
        ref_len = trajectory_length(reference, min_step=0.0)
    succ = is_success(traj, goal, d_th, stopped)
    tl = trajectory_length(traj)
    return {
        "success": succ,
        "oracle_success": oracle_success(traj, goal, d_th),
        "ne_m": navigation_error(traj, goal),
        "tl_m": tl,
        "ref_len_m": float(ref_len),
        "spl": spl(succ, ref_len, tl),
        "ndtw": ndtw(traj, reference, d_th, spacing=spacing),
        "sdtw": succ * ndtw(traj, reference, d_th, spacing=spacing),
        "n_points": int(len(_xy(traj))),
    }


if __name__ == "__main__":
    # tiny self-test: a perfect run along the reference should give nDTW≈1, SPL=1
    ref = [[0, 0], [1, 0], [2, 0], [2, 1]]
    goal = [2, 1]
    perfect = np.linspace([0, 0], [2, 1], 200)  # dense
    print("perfect:", compute_all(perfect, ref, goal, d_th=0.5))
    detour = np.array([[0, 0], [0, 2], [2, 2], [2, 1]])  # long way round
    print("detour :", compute_all(detour, ref, goal, d_th=0.5, stopped=True))