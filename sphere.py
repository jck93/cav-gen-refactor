from dataclasses import dataclass

import numpy as np


@dataclass
class Sphere:
    x: np.float64
    y: np.float64
    z: np.float64
    r: np.float64
    unit: str = "bohr"

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def convert_to_angstrom(self):
        if self.unit == "angstrom":
            return
        if self.unit == "bohr":
            bohr_to_angstrom = lambda bohr: bohr / 1.8897259886
            self.x = bohr_to_angstrom(self.x)
            self.y = bohr_to_angstrom(self.y)
            self.z = bohr_to_angstrom(self.z)
            self.r = bohr_to_angstrom(self.r)
            self.unit = "angstrom"


