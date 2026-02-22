import numpy as np

from .tessera_builder import TesseraBuilder
from .constants import *
from .sphere import Sphere
from .tessera import Tessera
from .cavity import Cavity
from .connectivity import Connectivity
from .get_vertex_positions import get_vertex_positions


def build_pcm_cavity(subtesserae_per_tessera, minimum_tessera_distance_angstrom, spheres, 
                     minimum_tessera_area_angstrom2=1e-8):
    for sphere in spheres:
        sphere.convert_to_angstrom()
    vertex_positions = get_vertex_positions()
    connectivity = Connectivity(subtesserae_per_tessera)
    cavity = Cavity()
    tessera_builder = TesseraBuilder(spheres)
    for isphere, sphere in enumerate(spheres):
        for itess in range(TESSERAE_PER_SPHERE):
            for isubtess in range(subtesserae_per_tessera):
                pts = np.zeros((PCM_DIM_SPACE, DIM_TEN), dtype=np.float64) # (1:PCM_DIM_SPACE, 1:DIM_TEN)
                pts[:, 0] = vertex_positions[connectivity.n0(itess, isubtess), [0, 2, 1]] * sphere.r + sphere.xyz
                pts[:, 1] = vertex_positions[connectivity.n1(itess, isubtess), [0, 2, 1]] * sphere.r + sphere.xyz
                pts[:, 2] = vertex_positions[connectivity.n2(itess, isubtess), [0, 2, 1]] * sphere.r + sphere.xyz
                tessera = tessera_builder.build(isphere, pts)
                if tessera.area >= minimum_tessera_area_angstrom2:
                    cavity.append(tessera)
    cavity.merge_close_tesserae(minimum_tessera_distance_angstrom)
    cavity.convert_to_bohr()
    return cavity

