from dataclasses import dataclass

import numpy as np


@dataclass
class Sphere:
    x: np.float64
    y: np.float64
    z: np.float64
    r: np.float64

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z], dtype=np.float64)

