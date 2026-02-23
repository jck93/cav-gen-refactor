import numpy as np

from .tessera_builder import TesseraBuilder
from .constants import TESSERAE_PER_SPHERE, MINIMUM_TESSERA_AREA_ANGSTROM2
from .sphere import Sphere
from .tessera import Tessera
from .cavity import Cavity
from .vertex_builder import VertexBuilder


def build_pcm_cavity(subtesserae_per_tessera, minimum_tessera_distance_angstrom, spheres): 
    for sphere in spheres:
        sphere.convert_to_angstrom()
    vertex_builder = VertexBuilder(spheres, subtesserae_per_tessera)
    tessera_builder = TesseraBuilder(spheres)
    cavity = Cavity()
    for isphere in range(len(spheres)):
        for itess in range(TESSERAE_PER_SPHERE):
            for isubtess in range(subtesserae_per_tessera):
                vertices = vertex_builder.build(isphere, itess, isubtess)
                tessera = tessera_builder.build(isphere, vertices)
                if tessera.area >= MINIMUM_TESSERA_AREA_ANGSTROM2:
                    cavity.append(tessera)
    cavity.merge_close_tesserae(minimum_tessera_distance_angstrom)
    cavity.convert_to_bohr()
    return cavity

