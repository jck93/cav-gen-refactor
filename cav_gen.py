from dataclasses import dataclass

import numpy as np

from .subtessera_refactored import subtessera
from .constants import *


def get_itesss_to_merge(tesserae, tess_min_dist):
    for itess, tess_i in enumerate(tesserae):
        if tess_i.area <= M_EPSILON:
            continue
        for jtess, tess_j in enumerate(tesserae[itess+1:], itess+1):
            if tess_j.area <= M_EPSILON:
                continue
            if np.linalg.norm(np.array(tess_i.point) - tess_j.point) < tess_min_dist:
                return itess, jtess
    return None


def merge(tesserae, itesss_to_merge):
    itess, jtess = itesss_to_merge
    tess_i, tess_j = tesserae[itess], tesserae[jtess]
    area = tess_i.area + tess_j.area
    point = np.array(tess_i.point) * tess_i.area + np.array(tess_j.point) * tess_j.area
    point /= area
    normal = tess_i.normal * tess_i.area + tess_j.normal * tess_j.area
    normal /= np.linalg.norm(normal)
    r_sphere = tess_i.r_sphere * tess_i.area + tess_j.r_sphere * tess_j.area
    r_sphere /= area
    tesserae[itess] = Tessera(point, normal, area, r_sphere)
    del tesserae[jtess]
    return tesserae




@dataclass
class Sphere:
    x: np.float64
    y: np.float64
    z: np.float64
    r: np.float64

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class Tessera:
    point: np.ndarray
    normal: np.ndarray
    area: np.float64
    r_sphere: np.float64


def cav_gen(tess_sphere, tess_min_distance, spheres):
    isfet = np.zeros(DIM_TEN*DIM_ANGLES, dtype=int)
    cv = np.zeros((DIM_VERTICES, PCM_DIM_SPACE), dtype=np.float64)
    xyz = np.zeros((tess_sphere * N_TESS_SPHERE, PCM_DIM_SPACE), dtype=np.float64)
    ast = np.zeros(tess_sphere * N_TESS_SPHERE, dtype=np.float64)
    nxyz = np.zeros((tess_sphere * N_TESS_SPHERE, PCM_DIM_SPACE), dtype=np.float64)

    rescaled_spheres = []
    to_angstrom = lambda bohr: bohr / 1.8897259886
    for sphere in spheres:
        rescaled_spheres.append(
            Sphere(
                to_angstrom(sphere.x),
                to_angstrom(sphere.y),
                to_angstrom(sphere.z),
                to_angstrom(sphere.r)
            )
        )
    spheres = rescaled_spheres

    cv[0, 2] = 1.0
    cv[121, 2] = -1.0

    index = 0
    for iangle in range(DIM_ANGLES):
        th = THEV[iangle]
        fi0 = FIV[iangle]
        cth = np.cos(th)
        sth = np.sin(th)
        for jangle in range(5):
            fi = fi0 + jangle * FIR
            index += 1
            cv[index, 0] = sth * np.cos(fi)
            cv[index, 1] = sth * np.sin(fi)
            cv[index, 2] = cth


    ntess = 0
    tesserae = []
    for isphere, sphere in enumerate(spheres):


        xyz[:] = 0.0
        ast[:] = 0
        nxyz[:] = 0


        for itess in range(N_TESS_SPHERE):
            for isubtess in range(tess_sphere):
                if tess_sphere == 1:
                    n1 = JVT1[0, itess]
                    n2 = JVT1[1, itess]
                    n3 = JVT1[2, itess]
                else:
                    if isubtess == 0:
                        n1 = JVT1[0, itess]
                        n2 = JVT1[4, itess]
                        n3 = JVT1[3, itess]
                    elif isubtess == 1:
                        n1 = JVT1[3, itess]
                        n2 = JVT1[5, itess]
                        n3 = JVT1[2, itess]
                    elif isubtess == 2:
                        n1 = JVT1[3, itess]
                        n2 = JVT1[4, itess]
                        n3 = JVT1[5, itess]
                    else:  # isubtess == 3
                        n1 = JVT1[1, itess]
                        n2 = JVT1[5, itess]
                        n3 = JVT1[4, itess]

                pts = np.zeros((PCM_DIM_SPACE, DIM_TEN), dtype=np.float64) # (1:PCM_DIM_SPACE, 1:DIM_TEN)

                pts[:, 0] = cv[n1, [0, 2, 1]] * sphere.r + sphere.xyz
                pts[:, 1] = cv[n2, [0, 2, 1]] * sphere.r + sphere.xyz
                pts[:, 2] = cv[n3, [0, 2, 1]] * sphere.r + sphere.xyz
                nv = 3
                pts, pp, pp1, area = subtessera(isphere, spheres, nv, pts)

                if abs(area) < M_EPSILON:
                    continue

                index = tess_sphere * itess + isubtess
                xyz[index] = pp.copy()
                nxyz[index] = pp1.copy() 
                ast[index] = area
                isfet[index] = isphere

        for itess in range(N_TESS_SPHERE * tess_sphere):
            if abs(ast[itess]) < M_EPSILON:
                continue
            tessera = Tessera(
                point=xyz[itess].copy(),
                normal=nxyz[itess].copy(),
                area=ast[itess],
                r_sphere=spheres[isfet[itess]].r
            )
            tesserae.append(tessera)

    ntess = len(tesserae)

    itesss_to_merge = get_itesss_to_merge(tesserae, tess_min_distance)
    while itesss_to_merge is not None:
        tesserae = merge(tesserae, itesss_to_merge)
        itesss_to_merge = get_itesss_to_merge(tesserae, tess_min_distance)
    volume = sum(tessera.area * np.dot(tessera.point, tessera.normal) for tessera in tesserae) / 3
    area = sum(tessera.area for tessera in tesserae)

    to_bohr = lambda angstrom: angstrom * 1.8897259886
    tesserae_bohr = []
    for tessera in tesserae:
        tesserae_bohr.append(
            Tessera(
                point=to_bohr(np.array(tessera.point)),
                normal=tessera.normal,
                area=tessera.area * 1.8897259886 ** 2,
                r_sphere=to_bohr(tessera.r_sphere)
            )
        )
    return tesserae_bohr


if __name__ == "__main__":
    spheres = [Sphere(0, 0, 0, 2), Sphere(1, 0, 0, 2), Sphere(0.5, 1.5, 0, 1.5)]

    tesserae = cav_gen(1, 0.3, spheres)
    for t in tesserae:
        print(t.point)

