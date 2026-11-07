
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class RobotConfig:
    L1: float = 100.0
    L2: float = 80.0
    L3: float = 40.0

    # None means no mechanical joint limits.
    q_min: tuple | None = None
    q_max: tuple | None = None

    def __post_init__(self):
        lengths = np.asarray(
            [self.L1, self.L2, self.L3],
            dtype=float
        )

        if not np.all(np.isfinite(lengths)):
            raise ValueError(
                "Robot link lengths must be finite."
            )

        if np.any(lengths <= 0):
            raise ValueError(
                "Robot link lengths must be positive."
            )

        if (self.q_min is None) != (self.q_max is None):
            raise ValueError(
                "q_min and q_max must both be provided or both be None."
            )

        if self.q_min is not None:
            q_min = np.asarray(self.q_min, dtype=float)
            q_max = np.asarray(self.q_max, dtype=float)

            if q_min.shape != (3,) or q_max.shape != (3,):
                raise ValueError(
                    "Joint limits must contain 3 values."
                )

            if not np.all(np.isfinite(q_min)) or not np.all(
                np.isfinite(q_max)
            ):
                raise ValueError(
                    "Joint limits must be finite."
                )

            if np.any(q_min >= q_max):
                raise ValueError(
                    "Each q_min must be smaller than q_max."
                )


@dataclass
class SolverConfig:
    position_tolerance: float = 1e-3
    orientation_tolerance: float = 1e-5

    max_iterations: int = 150
    step_gain: float = 1.0
    max_step_norm: float = 0.25

    damping: float = 0.05
    adaptive_sigma_threshold: float = 0.08

    # Convert orientation error (rad) to a length-like residual.
    orientation_weight: float = 40.0

    # Levenberg-Marquardt parameters
    lm_initial_lambda: float = 1e-2
    lm_min_lambda: float = 1e-9
    lm_max_lambda: float = 1e9

    def __post_init__(self):
        positive_values = {
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "max_step_norm": self.max_step_norm,
            "damping": self.damping,
            "adaptive_sigma_threshold": self.adaptive_sigma_threshold,
            "orientation_weight": self.orientation_weight,
            "lm_initial_lambda": self.lm_initial_lambda,
            "lm_min_lambda": self.lm_min_lambda,
            "lm_max_lambda": self.lm_max_lambda,
        }

        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive."
                )

        if self.max_iterations <= 0:
            raise ValueError(
                "max_iterations must be positive."
            )

        if not np.isfinite(self.step_gain) or not (
            0 < self.step_gain <= 1
        ):
            raise ValueError(
                "step_gain must be in the interval (0, 1]."
            )

        if not (
            self.lm_min_lambda
            <= self.lm_initial_lambda
            <= self.lm_max_lambda
        ):
            raise ValueError(
                "LM lambda parameters must satisfy "
                "lm_min_lambda <= lm_initial_lambda <= lm_max_lambda."
            )


@dataclass
class ExperimentConfig:
    duration: float = 8.0
    samples: int = 401

    orientation: float = 0.0

    shape_scale: float = 25.0
    shape_center: tuple = (175.0, 0.0)

    profiles: tuple = (
        "cubic",
        "quintic",
        "lspb"
    )

    methods: tuple = (
        "analytical",
        "pinv",
        "dls",
        "adaptive_dls",
        "lm"
    )

    scenarios: tuple = (
        "normal_circle",
        "normal_square",
        "normal_star",
        "near_singularity",
        "workspace_boundary",
        "unreachable"
    )

    output_dir: Path = Path("results")
    random_seed: int = 7

    def __post_init__(self):
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError(
                "duration must be finite and positive."
            )

        if self.samples < 2:
            raise ValueError(
                "samples must be at least 2."
            )

        if not np.isfinite(self.orientation):
            raise ValueError(
                "orientation must be finite."
            )

        if not np.isfinite(self.shape_scale) or self.shape_scale <= 0:
            raise ValueError(
                "shape_scale must be finite and positive."
            )

        center = np.asarray(
            self.shape_center,
            dtype=float
        )

        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError(
                "shape_center must contain two finite values."
            )

        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError(
                "Duplicate trajectory profiles are not allowed."
            )

        if len(set(self.methods)) != len(self.methods):
            raise ValueError(
                "Duplicate solver methods are not allowed."
            )

        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError(
                "Duplicate scenarios are not allowed."
            )

        self.output_dir = Path(self.output_dir)


ROBOT = RobotConfig()
SOLVER = SolverConfig()
EXPERIMENT = ExperimentConfig()
