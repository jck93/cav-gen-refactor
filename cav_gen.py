import numpy as np

from .subtessera import subtessera
from .constants import *
from .sphere import Sphere
from .tessera import Tessera, Tesserae
from .connectivity import Connectivity
from .get_vertex_positions import get_vertex_positions


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


def cav_gen(subtesserae_per_sphere, tess_min_distance, spheres):
    for sphere in spheres:
        sphere.convert_to_angstrom()
    vertex_positions = get_vertex_positions()
    connectivity = Connectivity(subtesserae_per_sphere)
    tesserae = Tesserae()
    for isphere, sphere in enumerate(spheres):
        for itess in range(TESSERAE_PER_SPHERE):
            for isubtess in range(subtesserae_per_sphere):
                pts = np.zeros((PCM_DIM_SPACE, DIM_TEN), dtype=np.float64) # (1:PCM_DIM_SPACE, 1:DIM_TEN)
                pts[:, 0] = vertex_positions[connectivity.n0(itess, isubtess), [0, 2, 1]] * sphere.r + sphere.xyz
                pts[:, 1] = vertex_positions[connectivity.n1(itess, isubtess), [0, 2, 1]] * sphere.r + sphere.xyz
                pts[:, 2] = vertex_positions[connectivity.n2(itess, isubtess), [0, 2, 1]] * sphere.r + sphere.xyz
                point, normal, area = subtessera(isphere, spheres, pts)
                if abs(area) >= M_EPSILON:
                    tesserae.append(Tessera(point, normal, area, sphere.r))
    tesserae.merge_close_tesserae(tess_min_distance)
    tesserae.convert_to_bohr()
    return tesserae

