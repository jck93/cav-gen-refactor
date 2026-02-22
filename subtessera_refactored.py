import numpy as np
from .inter import inter
from .gaubon import gaubon

def subtessera(isphere, spheres, nvertices, pts):
    """Compute uncovered tessera region for a sphere and its area, representative point, and normal."""
    
    TOL = -1e-10
    MAX_VERTICES = 10
    DIM_TEN = 10
    PCM_DIM_SPACE = 3

    sphere = spheres[isphere]
    intsph = isphere * np.ones(DIM_TEN, dtype=int)
    
    # Temporary arrays
    pscr = np.zeros((PCM_DIM_SPACE, DIM_TEN))
    cccp = np.zeros((PCM_DIM_SPACE, DIM_TEN))
    pointl = np.zeros((PCM_DIM_SPACE, DIM_TEN))
    ccc = np.zeros((PCM_DIM_SPACE, DIM_TEN))
    
    # Initialize ccc with sphere center
    ccc[:, :3] = np.array([sphere.x, sphere.y, sphere.z])[:, None]


    pp, pp1, area = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0

    # Loop over other spheres
    for jsphere, other in enumerate(spheres):
        if jsphere == isphere:
            continue

        intscr = intsph.copy()
        pscr[:, :nvertices] = pts[:, :nvertices]
        cccp[:, :nvertices] = ccc[:, :nvertices]

        # Determine which vertices are inside the other sphere
        ind = np.zeros(nvertices, dtype=int)
        for i in range(nvertices):
            vec = pts[:, i] - np.array([other.x, other.y, other.z])
            if np.linalg.norm(vec) < other.r:
                ind[i] = 1

        # If all vertices are inside the other sphere, tessera is fully covered
        if np.all(ind[:nvertices] == 1):
            return pts, pp, pp1, area

        # Determine edge types (ltyp)
        ltyp = np.zeros(nvertices, dtype=int)
        for i in range(nvertices):
            i_next = (i + 1) % nvertices
            if ind[i] == 1 and ind[i_next] == 1:
                ltyp[i] = 0
            elif ind[i] == 0 and ind[i_next] == 1:
                ltyp[i] = 1
            elif ind[i] == 1 and ind[i_next] == 0:
                ltyp[i] = 2
            elif ind[i] == 0 and ind[i_next] == 0:
                ltyp[i] = 4
                # Check for intersection along edge
                rc_vec = ccc[:, i] - pts[:, i]
                rc = np.linalg.norm(rc_vec)
                for j in range(1, 12):
                    point = pts[:, i] + j * (pts[:, i_next] - pts[:, i]) / 11
                    point -= ccc[:, i]
                    point = point * rc / np.linalg.norm(point) + ccc[:, i]
                    dist = np.linalg.norm(point - np.array([other.x, other.y, other.z]))
                    if abs(dist - other.r) < TOL:
                        ltyp[i] = 3
                        pointl[:, i] = point
                        break

        # Count edges that require cuts
        icut = np.sum((ltyp == 1) | (ltyp == 2)) + 2 * np.sum(ltyp == 3)
        icut //= 2
        if icut > 1:
            return pts, pp, pp1, area 


        # Build new vertices
        na = 0
        for i in range(nvertices):
            i_next = (i + 1) % nvertices

            if ltyp[i] == 0:
                continue
            elif ltyp[i] == 1:
                # Edge entering other sphere
                pts[:, na] = pscr[:, i]
                ccc[:, na] = cccp[:, i]
                intsph[na] = intscr[i]
                na += 1

                p4 = inter(other, pscr[:, i], pscr[:, i_next], cccp[:, i], 0)
                pts[:, na] = p4

                de2 = np.sum((np.array([other.x, other.y, other.z]) - np.array([sphere.x, sphere.y, sphere.z]))**2)
                prefactor = (sphere.r**2 - other.r**2 + de2) / (2 * de2)
                ccc[:, na] = np.array([sphere.x, sphere.y, sphere.z]) + prefactor * (np.array([other.x, other.y, other.z]) - np.array([sphere.x, sphere.y, sphere.z]))
                intsph[na] = jsphere
                na += 1

            elif ltyp[i] == 2:
                # Edge leaving other sphere
                p4 = inter(other, pscr[:, i], pscr[:, i_next], cccp[:, i], 1)
                pts[:, na] = p4
                ccc[:, na] = cccp[:, i]
                intsph[na] = intscr[i]
                na += 1

            elif ltyp[i] == 3:
                # Complex intersection along edge
                # First sub-edge
                pts[:, na] = pscr[:, i]
                ccc[:, na] = cccp[:, i]
                intsph[na] = intscr[i]
                na += 1

                p4 = inter(other, pscr[:, i], pointl[:, i], cccp[:, i], 0)
                pts[:, na] = p4

                de2 = np.sum((np.array([other.x, other.y, other.z]) - np.array([sphere.x, sphere.y, sphere.z]))**2)
                prefactor = (sphere.r**2 - other.r**2 + de2) / (2 * de2)
                ccc[:, na] = np.array([sphere.x, sphere.y, sphere.z]) + prefactor * (np.array([other.x, other.y, other.z]) - np.array([sphere.x, sphere.y, sphere.z]))
                intsph[na] = jsphere
                na += 1

                # Second sub-edge
                p4 = inter(other, pointl[:, i], pscr[:, i_next], cccp[:, i], 1)
                pts[:, na] = p4
                ccc[:, na] = cccp[:, i]
                intsph[na] = intscr[i]
                na += 1

            elif ltyp[i] == 4:
                # Edge fully outside
                pts[:, na] = pscr[:, i]
                ccc[:, na] = cccp[:, i]
                intsph[na] = intscr[i]
                na += 1

        nvertices = na
        if nvertices > MAX_VERTICES:
            raise Exception("Too many vertices on the tessera")

    # Compute area, representative point, and normal
    area, pp, pp1 = gaubon(spheres, nvertices, isphere, pts, ccc, intsph)


    return pts, pp, pp1, area

