import numpy as np

from .tessera_builder import TesseraBuilder
from .constants import TESSERAE_PER_SPHERE
from .sphere import Sphere
from .tessera import Tessera
from .cavity import Cavity
from .vertex_positions import VertexPositions


def build_pcm_cavity(subtesserae_per_tessera, minimum_tessera_distance_angstrom, spheres, 
                     minimum_tessera_area_angstrom2=1e-8):
    for sphere in spheres:
        sphere.convert_to_angstrom()
    vertex_positions = VertexPositions(subtesserae_per_tessera)
    cavity = Cavity()
    tessera_builder = TesseraBuilder(spheres)
    for isphere, sphere in enumerate(spheres):
        for itess in range(TESSERAE_PER_SPHERE):
            for isubtess in range(subtesserae_per_tessera):
                vertices = vertex_positions(sphere, itess, isubtess)
                tessera = tessera_builder.build(isphere, vertices)
                if tessera.area >= minimum_tessera_area_angstrom2:
                    cavity.append(tessera)
    cavity.merge_close_tesserae(minimum_tessera_distance_angstrom)
    cavity.convert_to_bohr()
    return cavity

