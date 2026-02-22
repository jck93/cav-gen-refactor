from dataclasses import dataclass

import numpy as np


@dataclass
class Tessera:
    point: np.ndarray
    normal: np.ndarray
    area: np.float64
    r_sphere: np.float64
    unit: str = "angstrom"

    def convert_to_bohr(self):
        if self.unit == "bohr":
            return
        if self.unit == "angstrom":
            from_angstrom_to_bohr = lambda angstrom: angstrom * 1.8897259886
            self.point = from_angstrom_to_bohr(self.point)
            self.area = from_angstrom_to_bohr(np.sqrt(self.area)) ** 2
            self.r_sphere = from_angstrom_to_bohr(self.r_sphere)
            self.unit = "bohr"


