
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


def solve_lm(
    robot,
    target,
    q0,
    cfg
):
    start = time.perf_counter()

    target = np.asarray(
        target,
        dtype=float
    ).reshape(-1)

    q = np.asarray(
        q0,
        dtype=float
    ).copy()

    if target.size != 3:
        raise ValueError(
            "Target must contain [x, y, phi]."
        )

    if not np.all(np.isfinite(target)):
        raise ValueError(
            "Target must contain only finite values."
        )

    if q.shape != (3,):
        raise ValueError(
            "q0 must contain 3 joint angles."
        )

    if not np.all(np.isfinite(q)):
        raise ValueError(
            "Initial joint angles must be finite."
        )

    lam = cfg.lm_initial_lambda
    nu = 2.0

    sigma_min = float("nan")

    # Check geometric reachability before IK
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

        H = Jw.T @ Jw
        g = Jw.T @ ew

        diagonal = np.maximum(
            np.diag(H),
            1e-12
        )

        D = np.diag(diagonal)

        A = H + lam * D

        try:
            dq = np.linalg.solve(
                A,
                g
            )

        except np.linalg.LinAlgError:
            lam = min(
                cfg.lm_max_lambda,
                lam * nu
            )

            nu *= 2.0
            continue

        dq = limit_step(
            dq,
            cfg.max_step_norm
        )

        candidate = q + dq

        f = 0.5 * np.dot(
            ew,
            ew
        )

        candidate_error = pose_error(
            target,
            robot.fk(candidate)
        )

        _, candidate_ew = weighted_system(
            robot,
            candidate,
            candidate_error,
            cfg.orientation_weight
        )

        f_new = 0.5 * np.dot(
            candidate_ew,
            candidate_ew
        )

        predicted = 0.5 * dq @ (
            lam * D @ dq + g
        )

        if predicted > 1e-18:
            rho = (f - f_new) / predicted
        else:
            rho = -1.0

        if rho > 0 and f_new < f:
            q = candidate

            lam = max(
                cfg.lm_min_lambda,
                lam * max(
                    1.0 / 3.0,
                    1.0 - (2.0 * rho - 1.0)**3
                )
            )

            nu = 2.0

        else:
            lam = min(
                cfg.lm_max_lambda,
                lam * nu
            )

            nu *= 2.0

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
