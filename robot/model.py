
from dataclasses import dataclass
import numpy as np


@dataclass
class Planar3R:

    L1: float = 100.0
    L2: float = 80.0
    L3: float = 40.0

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

        self.L1, self.L2, self.L3 = map(
            float,
            lengths
        )

    @staticmethod
    def _validate_q(q):
        q = np.asarray(q, dtype=float).reshape(-1)

        if q.size != 3:
            raise ValueError(
                "Joint configuration must contain 3 angles."
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "Joint angles must be finite."
            )

        return q

    @property
    def max_reach(self):
        return self.L1 + self.L2 + self.L3

    @property
    def min_reach(self):
        largest = max(
            self.L1,
            self.L2,
            self.L3
        )

        remaining = (
            self.L1 + self.L2 + self.L3 - largest
        )

        return max(
            0.0,
            largest - remaining
        )

    def fk(self, q):
        q = self._validate_q(q)

        a1 = q[0]
        a2 = q[0] + q[1]
        a3 = q[0] + q[1] + q[2]

        x = (
            self.L1 * np.cos(a1)
            + self.L2 * np.cos(a2)
            + self.L3 * np.cos(a3)
        )

        y = (
            self.L1 * np.sin(a1)
            + self.L2 * np.sin(a2)
            + self.L3 * np.sin(a3)
        )

        phi = a3

        return np.array(
            [x, y, phi],
            dtype=float
        )

    def points(self, q):
        q = self._validate_q(q)

        a1 = q[0]
        a2 = q[0] + q[1]
        a3 = q[0] + q[1] + q[2]

        p0 = np.array([0.0, 0.0])

        p1 = p0 + np.array([
            self.L1 * np.cos(a1),
            self.L1 * np.sin(a1)
        ])

        p2 = p1 + np.array([
            self.L2 * np.cos(a2),
            self.L2 * np.sin(a2)
        ])

        p3 = p2 + np.array([
            self.L3 * np.cos(a3),
            self.L3 * np.sin(a3)
        ])

        return np.vstack([
            p0,
            p1,
            p2,
            p3
        ])
