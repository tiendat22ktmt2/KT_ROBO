
import time
import numpy as np

from .common import (
    pose_error,
    weighted_system,
    finish_result,
    is_converged,
    limit_step,
    target_is_geometrically_reachable
)


def solve_adaptive_dls(
    robot,
    target,
    q0,
    cfg
):
    start = time.perf_counter()

    q = np.asarray(
        q0,
        dtype=float
    ).copy()

    target = np.asarray(
        target,
        dtype=float
    ).reshape(-1)

    sigma_min = float("nan")

    # Check target validity
    if target.size != 3:
        raise ValueError(
            "Target must contain [x, y, phi]."
        )

    if not np.all(np.isfinite(target)):
        raise ValueError(
            "Target must contain only finite values."
        )

    if not np.all(np.isfinite(q)):
        raise ValueError(
            "Initial joint angles must be finite."
        )

    # Check geometric reachability before starting IK
    if not target_is_geometrically_reachable(
        robot,
        target
    ):
        return finish_result(
            robot,
            target,
            q,
            False,
            0,
            start,
            sigma_min,
            "geometrically_unreachable"
        )

    for it in range(cfg.max_iterations + 1):

        current = robot.fk(q)

        error = pose_error(
            target,
            current
        )

        if is_converged(error, cfg):

            return finish_result(
                robot,
                target,
                q,
                True,
                it,
                start,
                sigma_min
            )

        if it == cfg.max_iterations:
            break

        Jw, ew = weighted_system(
            robot,
            q,
            error,
            cfg.orientation_weight
        )

        singular_values = np.linalg.svd(
            Jw,
            compute_uv=False
        )

        sigma_min = float(
            singular_values[-1]
        )

        ratio = (
            cfg.adaptive_sigma_threshold
            / max(sigma_min, 1e-12)
        )

        lam = cfg.damping * max(
            1.0,
            ratio
        )

        A = (
            Jw @ Jw.T
            + lam**2 * np.eye(3)
        )

        dq = (
            Jw.T
            @ np.linalg.solve(A, ew)
        )

        dq = limit_step(
            dq,
            cfg.max_step_norm
        )

        q += cfg.step_gain * dq

    return finish_result(
        robot,
        target,
        q,
        False,
        cfg.max_iterations,
        start,
        sigma_min,
        "max_iterations"
    )
