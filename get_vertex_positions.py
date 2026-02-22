import numpy as np

from .constants import DIM_VERTICES, PCM_DIM_SPACE, DIM_ANGLES


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

AZIMUTHAL_INCREMENT = 1.256637061


def get_vertex_positions():
    vertex_positions = np.zeros((DIM_VERTICES, PCM_DIM_SPACE), dtype=np.float64)
    vertex_positions[0, 2] = 1.0
    vertex_positions[121, 2] = -1.0
    index = 0
    for theta, phi_0 in zip(POLAR_ANGLES, AZIMUTHAL_ANGLES):
        for jangle in range(5):
            phi = phi_0 + jangle * AZIMUTHAL_INCREMENT
            index += 1
            vertex_positions[index, 0] = np.sin(theta) * np.cos(phi)
            vertex_positions[index, 1] = np.sin(theta) * np.sin(phi)
            vertex_positions[index, 2] = np.cos(theta)
    return vertex_positions






