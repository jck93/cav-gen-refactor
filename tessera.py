from dataclasses import dataclass

import numpy as np

from .constants import M_EPSILON


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


class Tesserae:
    def __init__(self):
        self._tesserae = []

    def append(self, tessera):
        self._tesserae.append(tessera)

    def __iter__(self):
        return iter(self._tesserae)
    
    def convert_to_bohr(self):
        for tessera in self._tesserae:
            tessera.convert_to_bohr()

    @property
    def volume(self):
        return sum(tessera.area * np.dot(tessera.point, tessera.normal) for tessera in self._tesserae) / 3
    
    @property
    def area(self):
        return sum(tessera.area for tessera in self._tesserae)

    def merge_close_tesserae(self, minimum_distance_angstrom):
        tesserae_to_merge = self._get_tesserae_to_merge(minimum_distance_angstrom)
        while tesserae_to_merge is not None:
            self._merge(*tesserae_to_merge)
            tesserae_to_merge = self._get_tesserae_to_merge(minimum_distance_angstrom)

    def _merge(self, itess, jtess):
        tess_i, tess_j = self._tesserae[itess], self._tesserae[jtess]
        area = tess_i.area + tess_j.area
        point = np.array(tess_i.point) * tess_i.area + np.array(tess_j.point) * tess_j.area
        point /= area
        normal = tess_i.normal * tess_i.area + tess_j.normal * tess_j.area
        normal /= np.linalg.norm(normal)
        r_sphere = tess_i.r_sphere * tess_i.area + tess_j.r_sphere * tess_j.area
        r_sphere /= area
        self._tesserae[itess] = Tessera(point, normal, area, r_sphere)
        del self._tesserae[jtess]

    def _get_tesserae_to_merge(self, minimum_distance_angstrom):
        for itess, tess_i in enumerate(self._tesserae):
            if tess_i.area <= M_EPSILON:
                continue
            for jtess, tess_j in enumerate(self._tesserae[itess+1:], itess+1):
                if tess_j.area <= M_EPSILON:
                    continue
                if np.linalg.norm(np.array(tess_i.point) - tess_j.point) < minimum_distance_angstrom:
                    return itess, jtess
        return None


