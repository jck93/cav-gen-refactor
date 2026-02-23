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
            return type(self)(self.x, self.y, self.z, self.z)
        if self.unit == "bohr":
            bohr_to_angstrom = lambda bohr: bohr / 1.8897259886
            x = bohr_to_angstrom(self.x)
            y = bohr_to_angstrom(self.y)
            z = bohr_to_angstrom(self.z)
            r = bohr_to_angstrom(self.r)
            unit = "angstrom"
            return type(self)(x, y, z, r, unit)


