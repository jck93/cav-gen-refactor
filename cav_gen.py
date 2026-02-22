from dataclasses import dataclass

import numpy as np

from .subtessera_refactored import subtessera



M_EPSILON = 1e-8


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


@dataclass
class Tessera:
    point: np.ndarray
    normal: np.ndarray
    area: np.float64
    r_sphere: np.float64


def cav_gen(tess_sphere, tess_min_distance, spheres):

    DIM_ANGLES = 24
    DIM_TEN = 10
    DIM_VERTICES = 122
    PCM_DIM_SPACE = 3
    MAX_VERTICES = 6
    MXTS = 10000
    N_TESS_SPHERE = 60


    thev = ( 
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

    fiv = (
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

    fir = 1.256637061

    isfet = np.zeros(DIM_TEN*DIM_ANGLES, dtype=int)

    idum = (
      1, 6, 2, 32, 36, 37, 1, 2, 3, 33, 32, 38, 1, 3, 4, 34,         
      33, 39, 1, 4, 5, 35, 34, 40, 1, 5, 6, 36, 35, 41, 7, 2, 6, 51, 
      42, 37, 8, 3, 2, 47, 43, 38, 9, 4, 3, 48, 44, 39, 10, 5, 4,    
      49, 45, 40, 11, 6, 5, 50, 46, 41, 8, 2, 12, 62, 47, 52, 9,     
      3, 13, 63, 48, 53, 10, 4, 14, 64, 49, 54, 11, 5, 15, 65, 50,   
      55, 7, 6, 16, 66, 51, 56, 7, 12, 2, 42, 57, 52, 8, 13, 3,      
      43, 58, 53, 9, 14, 4, 44, 59, 54, 10, 15, 5, 45, 60, 55, 11,   
      16, 6, 46, 61, 56, 8, 12, 18, 68, 62, 77, 9, 13, 19, 69, 63,   
      78, 10, 14, 20, 70, 64, 79, 11, 15, 21, 71, 65, 80, 7, 16,     
      17, 67, 66, 81, 7, 17, 12, 57, 67, 72, 8, 18, 13, 58, 68, 73,  
      9, 19, 14, 59, 69, 74, 10, 20, 15, 60, 70, 75, 11, 21, 16,     
      61, 71, 76, 22, 12, 17, 87, 82, 72, 23, 13, 18, 88, 83, 73,    
      24, 14, 19, 89, 84, 74, 25, 15, 20, 90, 85, 75, 26, 16, 21,    
      91, 86, 76, 22, 18, 12, 82, 92, 77, 23, 19, 13, 83, 93, 78,    
      24, 20, 14, 84, 94, 79, 25, 21, 15, 85, 95, 80, 26, 17, 16,    
      86, 96, 81, 22, 17, 27, 102, 87, 97, 23, 18, 28, 103, 88, 98,  
      24, 19, 29, 104, 89, 99, 25, 20, 30, 105, 90, 100, 26, 21,     
      31, 106, 91, 101, 22, 28, 18, 92, 107, 98, 23, 29, 19, 93,
      108, 99, 24, 30, 20, 94, 109, 100, 25, 31, 21, 95, 110, 101,   
      26, 27, 17, 96, 111, 97, 22, 27, 28, 107, 102, 112, 23, 28,    
      29, 108, 103, 113, 24, 29, 30, 109, 104, 114, 25, 30, 31,      
      110, 105, 115, 26, 31, 27, 111, 106, 116, 122, 28, 27, 117,    
      118, 112, 122, 29, 28, 118, 119, 113, 122, 30, 29, 119, 120,   
      114, 122, 31, 30, 120, 121, 115, 122, 27, 31, 121, 117, 116 
    )

    jvt1 = np.array(idum, dtype=int).reshape((6, 60), order="F") - 1

    cv = np.zeros((DIM_VERTICES, PCM_DIM_SPACE), dtype=np.float64)
    xctst = np.zeros(tess_sphere * N_TESS_SPHERE, dtype=np.float64)
    yctst = np.zeros(tess_sphere * N_TESS_SPHERE, dtype=np.float64)
    zctst = np.zeros(tess_sphere * N_TESS_SPHERE, dtype=np.float64)
    ast = np.zeros(tess_sphere * N_TESS_SPHERE, dtype=np.float64)
    nctst = np.zeros((PCM_DIM_SPACE, tess_sphere * N_TESS_SPHERE), dtype=np.float64)

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
        th = thev[iangle]
        fi0 = fiv[iangle]
        cth = np.cos(th)
        sth = np.sin(th)
        for jangle in range(5):
            fi = fi0 + jangle * fir
            index += 1
            cv[index, 0] = sth * np.cos(fi)
            cv[index, 1] = sth * np.sin(fi)
            cv[index, 2] = cth


    ntess = 0
    tesserae = []
    for isphere, sphere in enumerate(spheres):


        xctst[:] = 0
        yctst[:] = 0
        zctst[:] = 0
        ast[:] = 0
        nctst[:] = 0


        for itess in range(N_TESS_SPHERE):
            for isubtess in range(tess_sphere):
                if tess_sphere == 1:
                    n1 = jvt1[0, itess]
                    n2 = jvt1[1, itess]
                    n3 = jvt1[2, itess]
                else:
                    if isubtess == 0:
                        n1 = jvt1[0, itess]
                        n2 = jvt1[4, itess]
                        n3 = jvt1[3, itess]
                    elif isubtess == 1:
                        n1 = jvt1[3, itess]
                        n2 = jvt1[5, itess]
                        n3 = jvt1[2, itess]
                    elif isubtess == 2:
                        n1 = jvt1[3, itess]
                        n2 = jvt1[4, itess]
                        n3 = jvt1[5, itess]
                    else:  # isubtess == 3
                        n1 = jvt1[1, itess]
                        n2 = jvt1[5, itess]
                        n3 = jvt1[4, itess]

                pts = np.zeros((PCM_DIM_SPACE, DIM_TEN), dtype=np.float64) # (1:PCM_DIM_SPACE, 1:DIM_TEN)

                pts[0, 0] = cv[n1, 0] * sphere.r + sphere.x
                pts[1, 0] = cv[n1, 2] * sphere.r + sphere.y
                pts[2, 0] = cv[n1, 1] * sphere.r + sphere.z

                pts[0, 1] = cv[n2, 0] * sphere.r + sphere.x
                pts[1, 1] = cv[n2, 2] * sphere.r + sphere.y
                pts[2, 1] = cv[n2, 1] * sphere.r + sphere.z

                pts[0, 2] = cv[n3, 0] * sphere.r + sphere.x
                pts[1, 2] = cv[n3, 2] * sphere.r + sphere.y
                pts[2, 2] = cv[n3, 1] * sphere.r + sphere.z
                nv = 3
                pts, pp, pp1, area = subtessera(isphere, spheres, nv, pts)

                if abs(area) < M_EPSILON:
                    continue

                index = tess_sphere * itess + isubtess
                xctst[index] = pp[0]
                yctst[index] = pp[1]
                zctst[index] = pp[2]
                nctst[:, index] = pp1  # TODO: turn dimensions around
                ast[index] = area
                isfet[index] = isphere

        for itess in range(N_TESS_SPHERE * tess_sphere):
            if abs(ast[itess]) < M_EPSILON:
                continue
            tessera = Tessera(
                point=(xctst[itess], yctst[itess], zctst[itess]),
                normal=nctst[:, itess].copy(),
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

