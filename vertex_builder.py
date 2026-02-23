import numpy as np

from .connectivity import Connectivity
from .constants import PCM_DIM_SPACE, DIM_TEN


POLAR_ANGLES = ( 
  0.6523581398,  1.107148718, 1.382085796, 
   1.759506858,  2.034443936, 2.489234514,
  0.3261790699, 0.5535743589, 
  0.8559571251, 0.8559571251, 1.017221968,
   1.229116717,  1.229116717, 1.433327788,
   1.570796327,  1.570796327, 1.708264866,
   1.912475937,  1.912475937, 2.124370686,
   2.285635528,  2.285635528, 2.588018295,
   2.815413584
)
AZIMUTHAL_ANGLES = (
  0.6283185307, 0.0000000000,
  0.6283185307, 0.0000000000, 0.6283185307, 
  0.0000000000, 0.6283185307, 0.0000000000, 
  0.2520539002, 1.0045831610, 0.6283185307, 
  0.3293628477, 0.9272742138, 0.0000000000, 
  0.3141592654, 0.9424777961, 0.6283185307, 
  0.2989556830, 0.9576813784, 0.0000000000, 
  0.3762646305, 0.8803724309, 0.6283188307, 
  0.0000000000
)
AZIMUTHAL_STEPS = 5
AZIMUTHAL_INCREMENT = 2*np.pi / AZIMUTHAL_STEPS
NORTH_POLE = 0, 0, 1
SOUTH_POLE = 0, 0, -1


class VertexBuilder:
    def __init__(self, spheres, subtesserae_per_tessera):
        self._spheres = spheres
        self._connectivity = Connectivity(subtesserae_per_tessera)
        self._vertex_positions = self._get_vertex_positions()

    def build(self, isphere, itess, isubtess):
        sphere = self._spheres[isphere]
        vertices = np.zeros((PCM_DIM_SPACE, DIM_TEN), dtype=np.float64) # (1:PCM_DIM_SPACE, 1:DIM_TEN)
        unit_sphere_vertex_0 = self._vertex_positions[self._connectivity.n0(itess, isubtess), [0, 2, 1]]
        unit_sphere_vertex_1 = self._vertex_positions[self._connectivity.n1(itess, isubtess), [0, 2, 1]]
        unit_sphere_vertex_2 = self._vertex_positions[self._connectivity.n2(itess, isubtess), [0, 2, 1]]
        vertices[:, 0] = unit_sphere_vertex_0 * sphere.r + sphere.xyz
        vertices[:, 1] = unit_sphere_vertex_1 * sphere.r + sphere.xyz
        vertices[:, 2] = unit_sphere_vertex_2 * sphere.r + sphere.xyz
        return vertices

    def _get_vertex_positions(self):
        vertex_positions = [NORTH_POLE]
        for theta, phi_0 in zip(POLAR_ANGLES, AZIMUTHAL_ANGLES):
            for iphi in range(AZIMUTHAL_STEPS):
                phi = phi_0 + iphi * AZIMUTHAL_INCREMENT
                vertex_positions.append(
                    (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
                )
        vertex_positions.append(SOUTH_POLE)
        return np.array(vertex_positions, dtype=np.float64)


