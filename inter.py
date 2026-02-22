import numpy as np

def inter(sphere, p1, p2, p3, ia):
    TOL = 1e-8

    r = np.linalg.norm(p1 - p3)
    alpha = 0.5
    delta = 0.0
    m_iter = 1
    band_iter = False
    while not band_iter:
        if m_iter > 1000:
            raise Exception("Too many iterations in inter")
        band_iter = True
        alpha = alpha + delta
        p4 = p1 + alpha * (p2 - p1) - p3
        dnorm = np.linalg.norm(p4)
        p4 = p4 * r / dnorm + p3
        diff = np.linalg.norm((
            p4[0] - sphere.x,
            p4[1] - sphere.y,
            p4[2] - sphere.z
        )) - sphere.r
        if abs(diff) < TOL:
            return p4
        step = 1 / (2 ** (m_iter + 1))
        if ia == 0:
            if diff > 0: 
                delta = +step
            else:  # diff > 0: 
                delta = -step
        if ia == 1:
            if diff > 0: 
                delta = -step
            else:  # diff > 0: 
                delta = +step
        m_iter += 1
        band_iter = False


