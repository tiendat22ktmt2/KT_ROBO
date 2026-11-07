
from dataclasses import dataclass
import numpy as np
import time

from robot.jacobian import geometric_jacobian


def wrap_angle(angle):
    """Normalize angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose_error(target, current):
    """Calculate position and orientation error."""
    e = (
        np.asarray(target, dtype=float)
        - np.asarray(current, dtype=float)
    )

    e[2] = wrap_angle(e[2])
    return e


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    runtime_ms: float
    position_error: float
    orientation_error: float
    sigma_min: float
    reason: str = ""


def weighted_system(robot, q, error, orientation_weight):
    """
    Build weighted Jacobian and weighted error.

    Position is measured in mm.
    Orientation is measured in rad.
    """
    J = geometric_jacobian(robot, q).copy()

    W = np.diag([
        1.0,
        1.0,
        orientation_weight
    ])

    Jw = W @ J
    ew = W @ error

    return Jw, ew


def finish_result(
    robot,
    target,
    q,
    converged,
    iterations,
    start,
    sigma_min,
    reason=""
):
    current = robot.fk(q)
    e = pose_error(target, current)

    return IKResult(
        q=np.asarray(q, dtype=float).copy(),
        converged=bool(converged),
        iterations=int(iterations),
        runtime_ms=(
            time.perf_counter() - start
        ) * 1000.0,
        position_error=float(
            np.linalg.norm(e[:2])
        ),
        orientation_error=float(abs(e[2])),
        sigma_min=float(sigma_min),
        reason=reason
    )


def is_converged(error, cfg):
    position_ok = (
        np.linalg.norm(error[:2])
        <= cfg.position_tolerance
    )

    orientation_ok = (
        abs(error[2])
        <= cfg.orientation_tolerance
    )

    return position_ok and orientation_ok


def limit_step(dq, max_norm):
    norm = np.linalg.norm(dq)

    if norm > max_norm:
        dq = dq * (max_norm / norm)

    return dq


def target_is_geometrically_reachable(
    robot,
    target,
    tolerance=1e-9
):
    """
    Check geometric reachability of a planar 3R robot
    for a desired end-effector pose [x, y, phi].

    The wrist position is calculated by removing the
    third link from the desired end-effector position.

    Returns:
        bool: True if the target pose is geometrically reachable.
    """
    target = np.asarray(target, dtype=float).reshape(-1)

    if target.size != 3:
        raise ValueError(
            "Target must contain [x, y, phi]."
        )

    if not np.all(np.isfinite(target)):
        raise ValueError(
            "Target must contain only finite values."
        )

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "Tolerance must be finite and non-negative."
        )

    # Get robot link lengths
    if all(
        hasattr(robot, name)
        for name in ("L1", "L2", "L3")
    ):
        L1 = float(robot.L1)
        L2 = float(robot.L2)
        L3 = float(robot.L3)

    elif hasattr(robot, "L"):
        L1, L2, L3 = map(float, robot.L)

    else:
        raise AttributeError(
            "Robot must provide L1, L2, L3 or L."
        )

    lengths = np.array([L1, L2, L3], dtype=float)

    if not np.all(np.isfinite(lengths)):
        raise ValueError(
            "Robot link lengths must be finite."
        )

    if np.any(lengths <= 0):
        raise ValueError(
            "Robot link lengths must be positive."
        )

    x, y, phi = target

    # Calculate wrist position
    wrist_x = x - L3 * np.cos(phi)
    wrist_y = y - L3 * np.sin(phi)

    wrist_distance = np.hypot(
        wrist_x,
        wrist_y
    )

    # Reachability limits of the first two links
    min_reach = abs(L1 - L2)
    max_reach = L1 + L2

    return (
        min_reach - tolerance
        <= wrist_distance
        <= max_reach + tolerance
    )
