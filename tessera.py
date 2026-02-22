from dataclasses import dataclass

import numpy as np

@dataclass
class Tessera:
    point: np.ndarray
    normal: np.ndarray
    area: np.float64
    r_sphere: np.float64

