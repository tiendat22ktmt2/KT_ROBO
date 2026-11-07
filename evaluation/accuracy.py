import numpy as np


def position_errors(targets, achieved):
    """
    Calculate Euclidean position errors between target and achieved poses.

    Parameters
    ----------
    targets : array-like, shape (N, 3)
        Target poses [x, y, phi].
    achieved : array-like, shape (N, 3)
        Achieved poses [x, y, phi].

    Returns
    -------
    np.ndarray, shape (N,)
        Euclidean position errors in mm.
    """
    targets = np.asarray(targets, dtype=float)
    achieved = np.asarray(achieved, dtype=float)

    if targets.shape != achieved.shape:
        raise ValueError(
            f"targets and achieved must have the same shape. "
            f"Got {targets.shape} and {achieved.shape}."
        )

    if targets.ndim != 2 or targets.shape[1] < 2:
        raise ValueError(
            "targets and achieved must have shape (N, 2) or (N, 3)."
        )

    position_difference = targets[:, :2] - achieved[:, :2]

    return np.linalg.norm(position_difference, axis=1)


def orientation_errors(targets, achieved):
    """
    Calculate absolute wrapped orientation errors.

    Angular differences are wrapped to [-pi, pi] before taking
    their absolute values.

    Parameters
    ----------
    targets : array-like, shape (N, 3)
        Target poses [x, y, phi].
    achieved : array-like, shape (N, 3)
        Achieved poses [x, y, phi].

    Returns
    -------
    np.ndarray, shape (N,)
        Absolute orientation errors in radians.
    """
    targets = np.asarray(targets, dtype=float)
    achieved = np.asarray(achieved, dtype=float)

    if targets.shape != achieved.shape:
        raise ValueError(
            f"targets and achieved must have the same shape. "
            f"Got {targets.shape} and {achieved.shape}."
        )

    if targets.ndim != 2 or targets.shape[1] < 3:
        raise ValueError(
            "targets and achieved must have shape (N, 3)."
        )

    angle_difference = targets[:, 2] - achieved[:, 2]

    wrapped_difference = np.arctan2(
        np.sin(angle_difference),
        np.cos(angle_difference)
    )

    return np.abs(wrapped_difference)