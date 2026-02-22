import numpy as np


def gaubon(spheres, nvertices, isphere, pts, ccc, intsph):
    sphere = spheres[isphere]
    point_1 = np.zeros(3, dtype=np.float64)
    point_2 = np.zeros(3, dtype=np.float64)
    EPS = 1e-8
    sum1 = 0
    for ivertex in range(nvertices):
        point_1 = pts[:, ivertex] - ccc[:, ivertex]
        point_2 = pts[:, 0 if (ivertex == nvertices - 1) else (ivertex+1)] - ccc[:, ivertex]
        dnorm1 = np.linalg.norm(point_1)
        dnorm2 = np.linalg.norm(point_2)
        cosphin = np.dot(point_1, point_2) / (dnorm1 * dnorm2)
        if cosphin > 1:
            cosphin = 1.0
        if cosphin < -1:
            cosphin = -1.0
        phin = np.arccos(cosphin)
        jsphere = intsph[ivertex]
        point_1[0] = spheres[jsphere].x - sphere.x
        point_1[1] = spheres[jsphere].y - sphere.y
        point_1[2] = spheres[jsphere].z - sphere.z
        dnorm1 = np.linalg.norm(point_1)
        if abs(dnorm1) < EPS:
            dnorm1 = 1.0
        point_2[0] = pts[0, ivertex] - sphere.x
        point_2[1] = pts[1, ivertex] - sphere.y
        point_2[2] = pts[2, ivertex] - sphere.z
        dnorm2 = np.linalg.norm(point_2)
        costn = np.dot(point_1, point_2) / (dnorm1 * dnorm2)
        sum1 += phin * costn
    sum2 = 0
    for ivertex in range(nvertices):
        n1 = ivertex
        if (ivertex > 0):
            n0 = ivertex - 1
        if (ivertex == 0):
            n0 = nvertices - 1
        if (ivertex < nvertices - 1):
            n2 = ivertex + 1
        if (ivertex == nvertices - 1):
            n2 = 0
        p1 = pts[:, n1] - ccc[:, n0]
        p2 = pts[:, n0] - ccc[:, n0]
        p3 = np.cross(p1, p2)
        p2 = p3

        p3 = np.cross(p1, p2)
        u1 = p3 / np.linalg.norm(p3)

        p1 = pts[:, n1] - ccc[:, n1]
        p2 = pts[:, n2] - ccc[:, n1]
        p3 = np.cross(p1, p2)
        p2 = p3

        p3 = np.cross(p1, p2)
        u2 = p3 / np.linalg.norm(p3)



        dot = np.dot(u1, u2)
        dot = max(-1.0, min(1.0, dot))
        betan = np.arccos(dot)


        sum2 += np.pi - betan
    area = sphere.r ** 2 * (2 * np.pi + sum1 - sum2)
    pp = np.zeros(3, dtype=np.float64)
    for ivertex in range(nvertices):
        pp[0] += pts[0, ivertex] - sphere.x
        pp[1] += pts[1, ivertex] - sphere.y
        pp[2] += pts[2, ivertex] - sphere.z
    dnorm = np.linalg.norm(pp)
    pp[0] = sphere.x + pp[0] * sphere.r / dnorm
    pp[1] = sphere.y + pp[1] * sphere.r / dnorm
    pp[2] = sphere.z + pp[2] * sphere.r / dnorm

    pp1 = np.zeros(3, dtype=np.float64)
    pp1[0] = (pp[0] - sphere.x) / sphere.r
    pp1[1] = (pp[1] - sphere.y) / sphere.r
    pp1[2] = (pp[2] - sphere.z) / sphere.r
    if area < 0:
        area = 0
    return area, pp, pp1

