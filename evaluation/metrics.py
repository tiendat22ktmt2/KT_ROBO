import numpy as np
import pandas as pd

from evaluation.accuracy import (
    position_errors,
    orientation_errors
)

from evaluation.performance import time_derivatives


def _safe_rmse(errors):
    """Calculate RMSE safely for an empty or non-finite array."""
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    if errors.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(errors ** 2)))


def _safe_max(errors):
    """Calculate maximum error safely."""
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    if errors.size == 0:
        return np.nan

    return float(np.max(errors))


def _safe_mean(values):
    """Calculate mean safely."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.mean(values))


def _safe_min(values):
    """Calculate minimum safely."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.min(values))


def evaluate_run(t, targets, achieved, q, results, feasible_mask):
    """
    Evaluate one IK simulation run.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time samples in seconds.

    targets : array-like, shape (N, 3)
        Desired poses [x, y, phi].

    achieved : array-like, shape (N, 3)
        Actual poses obtained from forward kinematics.

    q : array-like, shape (N, 3)
        Joint configurations in radians.

    results : list
        List of IKResult objects.

    feasible_mask : array-like, shape (N,)
        True if the corresponding target is geometrically reachable.

    Returns
    -------
    metrics : dict
        Summary evaluation metrics.

    frame : pandas.DataFrame
        Per-sample evaluation results.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate input data
    # ---------------------------------------------------------

    t = np.asarray(t, dtype=float)
    targets = np.asarray(targets, dtype=float)
    achieved = np.asarray(achieved, dtype=float)
    q = np.asarray(q, dtype=float)
    feasible_mask = np.asarray(feasible_mask, dtype=bool)

    n_samples = len(t)

    if n_samples == 0:
        raise ValueError("Cannot evaluate an empty simulation.")

    if targets.shape != (n_samples, 3):
        raise ValueError(
            f"targets must have shape ({n_samples}, 3), "
            f"got {targets.shape}."
        )

    if achieved.shape != (n_samples, 3):
        raise ValueError(
            f"achieved must have shape ({n_samples}, 3), "
            f"got {achieved.shape}."
        )

    if q.shape != (n_samples, 3):
        raise ValueError(
            f"q must have shape ({n_samples}, 3), "
            f"got {q.shape}."
        )

    if feasible_mask.shape != (n_samples,):
        raise ValueError(
            f"feasible_mask must have shape ({n_samples},), "
            f"got {feasible_mask.shape}."
        )

    if len(results) != n_samples:
        raise ValueError(
            f"results must contain {n_samples} entries, "
            f"got {len(results)}."
        )

    # ---------------------------------------------------------
    # 2. Calculate position and orientation errors
    # ---------------------------------------------------------

    pos_errors = position_errors(targets, achieved)
    ori_errors = orientation_errors(targets, achieved)

    # Only geometrically reachable targets are used for
    # feasible-only accuracy evaluation.
    feasible_errors = pos_errors[feasible_mask]
    feasible_ori_errors = ori_errors[feasible_mask]

    # ---------------------------------------------------------
    # 3. Extract solver information
    # ---------------------------------------------------------

    converged = np.array(
        [bool(result.converged) for result in results],
        dtype=bool
    )

    iterations = np.array(
        [result.iterations for result in results],
        dtype=float
    )

    solver_times = np.array(
        [result.runtime_ms for result in results],
        dtype=float
    )

    sigma_values = np.array(
        [
            result.sigma_min
            if result.sigma_min is not None
            else np.nan
            for result in results
        ],
        dtype=float
    )

    failure_reasons = np.array(
        [
            result.reason if result.reason is not None else ""
            for result in results
        ],
        dtype=object
    )

    # ---------------------------------------------------------
    # 4. Calculate convergence statistics
    # ---------------------------------------------------------

    feasible_count = int(np.sum(feasible_mask))
    unreachable_count = int(n_samples - feasible_count)

    successful_count = int(
        np.sum(converged & feasible_mask)
    )

    failed_feasible_count = int(
        np.sum((~converged) & feasible_mask)
    )

    if feasible_count > 0:
        convergence_rate = (
            successful_count / feasible_count
        ) * 100.0
    else:
        # Undefined when no target is geometrically reachable.
        convergence_rate = np.nan

    # ---------------------------------------------------------
    # 5. Calculate unreachable detection statistics
    # ---------------------------------------------------------

    # A target is considered correctly detected as unreachable
    # when it is geometrically unreachable and the solver reports
    # the corresponding failure reason.
    unreachable_detected = (
        (~feasible_mask)
        & (failure_reasons == "geometrically_unreachable")
    )

    detected_unreachable_count = int(
        np.sum(unreachable_detected)
    )

    if unreachable_count > 0:
        unreachable_detection_rate = (
            detected_unreachable_count / unreachable_count
        ) * 100.0
    else:
        # Undefined when the trajectory has no unreachable targets.
        unreachable_detection_rate = np.nan

    # ---------------------------------------------------------
    # 6. Calculate joint velocity, acceleration and jerk
    # ---------------------------------------------------------

    # Only converged and geometrically reachable samples
    # are considered valid for motion analysis.
    valid_motion = converged & feasible_mask

    velocity = np.full_like(q, np.nan, dtype=float)
    acceleration = np.full_like(q, np.nan, dtype=float)
    jerk = np.full_like(q, np.nan, dtype=float)

    # Split valid samples into consecutive segments.
    valid_indices = np.flatnonzero(valid_motion)

    if valid_indices.size >= 4:
        segments = np.split(
            valid_indices,
            np.where(np.diff(valid_indices) != 1)[0] + 1
        )

        for segment in segments:
            if len(segment) < 4:
                continue

            v, a, j = time_derivatives(
                t[segment],
                q[segment]
            )

            velocity[segment] = v
            acceleration[segment] = a
            jerk[segment] = j

    max_joint_speed = _safe_max(np.abs(velocity))
    max_joint_accel = _safe_max(np.abs(acceleration))
    max_joint_jerk = _safe_max(np.abs(jerk))

    # ---------------------------------------------------------
    # 7. Calculate summary metrics
    # ---------------------------------------------------------

    metrics = {
        # Overall accuracy: includes reachable and unreachable targets.
        "position_rmse_mm": _safe_rmse(pos_errors),
        "position_max_error_mm": _safe_max(pos_errors),

        # Accuracy on geometrically reachable targets only.
        "feasible_position_rmse_mm": _safe_rmse(
            feasible_errors
        ),
        "feasible_position_max_error_mm": _safe_max(
            feasible_errors
        ),

        "orientation_rmse_rad": _safe_rmse(ori_errors),
        "feasible_orientation_rmse_rad": _safe_rmse(
            feasible_ori_errors
        ),

        # Convergence statistics.
        "convergence_rate_pct": float(convergence_rate),
        "feasible_target_count": feasible_count,
        "successful_target_count": successful_count,

        # Unreachable target statistics.
        "unreachable_target_count": unreachable_count,
        "detected_unreachable_target_count": (
            detected_unreachable_count
        ),
        "unreachable_detection_rate_pct": float(
            unreachable_detection_rate
        ),

        # Solver failures on geometrically reachable targets.
        "failed_feasible_target_count": failed_feasible_count,

        # Solver performance on geometrically reachable targets.
        "mean_iterations": _safe_mean(
            iterations[feasible_mask]
        ),
        "mean_solver_time_ms": _safe_mean(
            solver_times[feasible_mask]
        ),

        # Robot motion performance.
        "max_joint_speed_rad_s": max_joint_speed,
        "max_joint_accel_rad_s2": max_joint_accel,
        "max_joint_jerk_rad_s3": max_joint_jerk,

        # Manipulability / singularity indicator.
        "min_sigma_value": _safe_min(sigma_values),
    }

    # ---------------------------------------------------------
    # 8. Create per-sample DataFrame
    # ---------------------------------------------------------

    frame = pd.DataFrame({
        "time_s": t,

        "target_x_mm": targets[:, 0],
        "target_y_mm": targets[:, 1],
        "target_phi_rad": targets[:, 2],

        "actual_x_mm": achieved[:, 0],
        "actual_y_mm": achieved[:, 1],
        "actual_phi_rad": achieved[:, 2],

        "position_error_mm": pos_errors,
        "orientation_error_rad": ori_errors,

        "q1_rad": q[:, 0],
        "q2_rad": q[:, 1],
        "q3_rad": q[:, 2],

        "q1_velocity_rad_s": velocity[:, 0],
        "q2_velocity_rad_s": velocity[:, 1],
        "q3_velocity_rad_s": velocity[:, 2],

        "q1_accel_rad_s2": acceleration[:, 0],
        "q2_accel_rad_s2": acceleration[:, 1],
        "q3_accel_rad_s2": acceleration[:, 2],

        "q1_jerk_rad_s3": jerk[:, 0],
        "q2_jerk_rad_s3": jerk[:, 1],
        "q3_jerk_rad_s3": jerk[:, 2],

        "converged": converged,
        "feasible": feasible_mask,

        "iterations": iterations,
        "solver_time_ms": solver_times,
        "sigma_min": sigma_values,

        "failure_reason": failure_reasons,
    })

    return metrics, frame