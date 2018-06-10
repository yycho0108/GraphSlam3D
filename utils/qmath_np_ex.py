import numpy as np

def Aij(p0,p1,dp,q0,q1,dq):
    pxi,pyi,pzi = p0
    pxj,pyj,pzj = p1
    pxij, pyij, pzij = dp
    qxi,qyi,qzi,qwi = q0
    qxj,qyj,qzj,qwj = q1
    qxij,qyij,qzij,qwij = dq

    res = [[(-2*qxi*qyi + 2*qwi*qzi)*(2*qxij*qyij + 2*qwij*qzij) + 
        (-2*qwi*qyi - 2*qxi*qzi)*(-2*qwij*qyij + 2*qxij*qzij) + 
        (-1 + 2*qyi**2 + 2*qzi**2)*(1 - 2*qyij**2 - 2*qzij**2),
        (-1 + 2*qxi**2 + 2*qzi**2)*(2*qxij*qyij + 2*qwij*qzij) + 
        (2*qwi*qxi - 2*qyi*qzi)*(-2*qwij*qyij + 2*qxij*qzij) + 
        (-2*qxi*qyi - 2*qwi*qzi)*(1 - 2*qyij**2 - 2*qzij**2),
        (-2*qwi*qxi - 2*qyi*qzi)*(2*qxij*qyij + 2*qwij*qzij) + 
        (-1 + 2*qxi**2 + 2*qyi**2)*(-2*qwij*qyij + 2*qxij*qzij) + 
        (2*qwi*qyi - 2*qxi*qzi)*(1 - 2*qyij**2 - 2*qzij**2),
        (2*(-pzi + pzj)*qwi - 4*(-pyi + pyj)*qxi + 2*(-pxi + pxj)*qyi)*
        (2*qxij*qyij + 2*qwij*qzij) + 
        (-2*(-pyi + pyj)*qwi - 4*(-pzi + pzj)*qxi + 2*(-pxi + pxj)*qzi)*
        (-2*qwij*qyij + 2*qxij*qzij) + 
        (2*(-pyi + pyj)*qyi + 2*(-pzi + pzj)*qzi)*(1 - 2*qyij**2 - 2*qzij**2),
        (2*(-pxi + pxj)*qxi + 2*(-pzi + pzj)*qzi)*(2*qxij*qyij + 2*qwij*qzij) + 
        (2*(-pxi + pxj)*qwi - 4*(-pzi + pzj)*qyi + 2*(-pyi + pyj)*qzi)*
        (-2*qwij*qyij + 2*qxij*qzij) + 
        (-2*(-pzi + pzj)*qwi + 2*(-pyi + pyj)*qxi - 4*(-pxi + pxj)*qyi)*
        (1 - 2*qyij**2 - 2*qzij**2),
        (-2*(-pxi + pxj)*qwi + 2*(-pzi + pzj)*qyi - 4*(-pyi + pyj)*qzi)*
        (2*qxij*qyij + 2*qwij*qzij) + 
        (2*(-pxi + pxj)*qxi + 2*(-pyi + pyj)*qyi)*(-2*qwij*qyij + 2*qxij*qzij) + 
        (2*(-pyi + pyj)*qwi + 2*(-pzi + pzj)*qxi - 4*(-pxi + pxj)*qzi)*
        (1 - 2*qyij**2 - 2*qzij**2),
        (2*(-pzi + pzj)*qxi - 2*(-pxi + pxj)*qzi)*(2*qxij*qyij + 2*qwij*qzij) + 
        (-2*(-pyi + pyj)*qxi + 2*(-pxi + pxj)*qyi)*(-2*qwij*qyij + 2*qxij*qzij) + 
        (-2*(-pzi + pzj)*qyi + 2*(-pyi + pyj)*qzi)*(1 - 2*qyij**2 - 2*qzij**2)],
        [(-1 + 2*qyi**2 + 2*qzi**2)*(2*qxij*qyij - 2*qwij*qzij) + 
            (-2*qwi*qyi - 2*qxi*qzi)*(2*qwij*qxij + 2*qyij*qzij) + 
            (-2*qxi*qyi + 2*qwi*qzi)*(1 - 2*qxij**2 - 2*qzij**2),
            (-2*qxi*qyi - 2*qwi*qzi)*(2*qxij*qyij - 2*qwij*qzij) + 
            (2*qwi*qxi - 2*qyi*qzi)*(2*qwij*qxij + 2*qyij*qzij) + 
            (-1 + 2*qxi**2 + 2*qzi**2)*(1 - 2*qxij**2 - 2*qzij**2),
            (2*qwi*qyi - 2*qxi*qzi)*(2*qxij*qyij - 2*qwij*qzij) + 
            (-1 + 2*qxi**2 + 2*qyi**2)*(2*qwij*qxij + 2*qyij*qzij) + 
            (-2*qwi*qxi - 2*qyi*qzi)*(1 - 2*qxij**2 - 2*qzij**2),
            (2*(-pyi + pyj)*qyi + 2*(-pzi + pzj)*qzi)*(2*qxij*qyij - 2*qwij*qzij) + 
            (-2*(-pyi + pyj)*qwi - 4*(-pzi + pzj)*qxi + 2*(-pxi + pxj)*qzi)*
            (2*qwij*qxij + 2*qyij*qzij) + 
            (2*(-pzi + pzj)*qwi - 4*(-pyi + pyj)*qxi + 2*(-pxi + pxj)*qyi)*
            (1 - 2*qxij**2 - 2*qzij**2),
            (-2*(-pzi + pzj)*qwi + 2*(-pyi + pyj)*qxi - 4*(-pxi + pxj)*qyi)*
            (2*qxij*qyij - 2*qwij*qzij) + 
            (2*(-pxi + pxj)*qwi - 4*(-pzi + pzj)*qyi + 2*(-pyi + pyj)*qzi)*
            (2*qwij*qxij + 2*qyij*qzij) + 
            (2*(-pxi + pxj)*qxi + 2*(-pzi + pzj)*qzi)*(1 - 2*qxij**2 - 2*qzij**2),
            (2*(-pyi + pyj)*qwi + 2*(-pzi + pzj)*qxi - 4*(-pxi + pxj)*qzi)*
            (2*qxij*qyij - 2*qwij*qzij) + 
            (2*(-pxi + pxj)*qxi + 2*(-pyi + pyj)*qyi)*(2*qwij*qxij + 2*qyij*qzij) + 
            (-2*(-pxi + pxj)*qwi + 2*(-pzi + pzj)*qyi - 4*(-pyi + pyj)*qzi)*
            (1 - 2*qxij**2 - 2*qzij**2),
            (-2*(-pzi + pzj)*qyi + 2*(-pyi + pyj)*qzi)*(2*qxij*qyij - 2*qwij*qzij) + 
            (-2*(-pyi + pyj)*qxi + 2*(-pxi + pxj)*qyi)*(2*qwij*qxij + 2*qyij*qzij) + 
            (2*(-pzi + pzj)*qxi - 2*(-pxi + pxj)*qzi)*(1 - 2*qxij**2 - 2*qzij**2)],
        [(1 - 2*qxij**2 - 2*qyij**2)*(-2*qwi*qyi - 2*qxi*qzi) + 
                (-1 + 2*qyi**2 + 2*qzi**2)*(2*qwij*qyij + 2*qxij*qzij) + 
                (-2*qxi*qyi + 2*qwi*qzi)*(-2*qwij*qxij + 2*qyij*qzij),
                (1 - 2*qxij**2 - 2*qyij**2)*(2*qwi*qxi - 2*qyi*qzi) + 
                (-2*qxi*qyi - 2*qwi*qzi)*(2*qwij*qyij + 2*qxij*qzij) + 
                (-1 + 2*qxi**2 + 2*qzi**2)*(-2*qwij*qxij + 2*qyij*qzij),
                (-1 + 2*qxi**2 + 2*qyi**2)*(1 - 2*qxij**2 - 2*qyij**2) + 
                (2*qwi*qyi - 2*qxi*qzi)*(2*qwij*qyij + 2*qxij*qzij) + 
                (-2*qwi*qxi - 2*qyi*qzi)*(-2*qwij*qxij + 2*qyij*qzij),
                (1 - 2*qxij**2 - 2*qyij**2)*
                (-2*(-pyi + pyj)*qwi - 4*(-pzi + pzj)*qxi + 2*(-pxi + pxj)*qzi) + 
                (2*(-pyi + pyj)*qyi + 2*(-pzi + pzj)*qzi)*(2*qwij*qyij + 2*qxij*qzij) + 
                (2*(-pzi + pzj)*qwi - 4*(-pyi + pyj)*qxi + 2*(-pxi + pxj)*qyi)*
                (-2*qwij*qxij + 2*qyij*qzij),
                (1 - 2*qxij**2 - 2*qyij**2)*
                (2*(-pxi + pxj)*qwi - 4*(-pzi + pzj)*qyi + 2*(-pyi + pyj)*qzi) + 
                (-2*(-pzi + pzj)*qwi + 2*(-pyi + pyj)*qxi - 4*(-pxi + pxj)*qyi)*
                (2*qwij*qyij + 2*qxij*qzij) + 
                (2*(-pxi + pxj)*qxi + 2*(-pzi + pzj)*qzi)*(-2*qwij*qxij + 2*qyij*qzij),
                (2*(-pxi + pxj)*qxi + 2*(-pyi + pyj)*qyi)*(1 - 2*qxij**2 - 2*qyij**2) + 
                (2*(-pyi + pyj)*qwi + 2*(-pzi + pzj)*qxi - 4*(-pxi + pxj)*qzi)*
                (2*qwij*qyij + 2*qxij*qzij) + 
                (-2*(-pxi + pxj)*qwi + 2*(-pzi + pzj)*qyi - 4*(-pyi + pyj)*qzi)*
                (-2*qwij*qxij + 2*qyij*qzij),
                (-2*(-pyi + pyj)*qxi + 2*(-pxi + pxj)*qyi)*(1 - 2*qxij**2 - 2*qyij**2) + 
                (-2*(-pzi + pzj)*qyi + 2*(-pyi + pyj)*qzi)*(2*qwij*qyij + 2*qxij*qzij) + 
                (2*(-pzi + pzj)*qxi - 2*(-pxi + pxj)*qzi)*(-2*qwij*qxij + 2*qyij*qzij)],
        [0,0,0,-(qwij*qwj) - qxij*qxj + qyij*qyj + qzij*qzj,
                -(qxj*qyij) - qxij*qyj - qwj*qzij - qwij*qzj,
                qwj*qyij + qwij*qyj - qxj*qzij - qxij*qzj,
                -(qwj*qxij) + qwij*qxj + qyj*qzij - qyij*qzj],
        [0,0,0,-(qxj*qyij) - qxij*qyj + qwj*qzij + qwij*qzj,
                -(qwij*qwj) + qxij*qxj - qyij*qyj + qzij*qzj,
                -(qwj*qxij) - qwij*qxj - qyj*qzij - qyij*qzj,
                -(qwj*qyij) + qwij*qyj - qxj*qzij + qxij*qzj],
        [0,0,0,-(qwj*qyij) - qwij*qyj - qxj*qzij - qxij*qzj,
                qwj*qxij + qwij*qxj - qyj*qzij - qyij*qzj,
                -(qwij*qwj) + qxij*qxj + qyij*qyj - qzij*qzj,
                qxj*qyij - qxij*qyj - qwj*qzij + qwij*qzj]]
    return np.asarray(res, dtype=np.float64)

def Bij(p0,p1,dp,q0,q1,dq):

    pxi,pyi,pzi = p0
    pxj,pyj,pzj = p1
    pxij, pyij, pzij = dp
    qxi,qyi,qzi,qwi = q0
    qxj,qyj,qzj,qwj = q1
    qxij,qyij,qzij,qwij = dq

    res = [[(2*qxi*qyi - 2*qwi*qzi)*(2*qxij*qyij + 2*qwij*qzij) + 
        (2*qwi*qyi + 2*qxi*qzi)*(-2*qwij*qyij + 2*qxij*qzij) + 
        (1 - 2*qyi**2 - 2*qzi**2)*(1 - 2*qyij**2 - 2*qzij**2),
        (1 - 2*qxi**2 - 2*qzi**2)*(2*qxij*qyij + 2*qwij*qzij) + 
        (-2*qwi*qxi + 2*qyi*qzi)*(-2*qwij*qyij + 2*qxij*qzij) + 
        (2*qxi*qyi + 2*qwi*qzi)*(1 - 2*qyij**2 - 2*qzij**2),
        (2*qwi*qxi + 2*qyi*qzi)*(2*qxij*qyij + 2*qwij*qzij) + 
        (1 - 2*qxi**2 - 2*qyi**2)*(-2*qwij*qyij + 2*qxij*qzij) + 
        (-2*qwi*qyi + 2*qxi*qzi)*(1 - 2*qyij**2 - 2*qzij**2),0,0,0,0],
        [(1 - 2*qyi**2 - 2*qzi**2)*(2*qxij*qyij - 2*qwij*qzij) + 
            (2*qwi*qyi + 2*qxi*qzi)*(2*qwij*qxij + 2*qyij*qzij) + 
            (2*qxi*qyi - 2*qwi*qzi)*(1 - 2*qxij**2 - 2*qzij**2),
            (2*qxi*qyi + 2*qwi*qzi)*(2*qxij*qyij - 2*qwij*qzij) + 
            (-2*qwi*qxi + 2*qyi*qzi)*(2*qwij*qxij + 2*qyij*qzij) + 
            (1 - 2*qxi**2 - 2*qzi**2)*(1 - 2*qxij**2 - 2*qzij**2),
            (-2*qwi*qyi + 2*qxi*qzi)*(2*qxij*qyij - 2*qwij*qzij) + 
            (1 - 2*qxi**2 - 2*qyi**2)*(2*qwij*qxij + 2*qyij*qzij) + 
            (2*qwi*qxi + 2*qyi*qzi)*(1 - 2*qxij**2 - 2*qzij**2),0,0,0,0],
        [(1 - 2*qxij**2 - 2*qyij**2)*(2*qwi*qyi + 2*qxi*qzi) + 
            (1 - 2*qyi**2 - 2*qzi**2)*(2*qwij*qyij + 2*qxij*qzij) + 
            (2*qxi*qyi - 2*qwi*qzi)*(-2*qwij*qxij + 2*qyij*qzij),
            (1 - 2*qxij**2 - 2*qyij**2)*(-2*qwi*qxi + 2*qyi*qzi) + 
            (2*qxi*qyi + 2*qwi*qzi)*(2*qwij*qyij + 2*qxij*qzij) + 
            (1 - 2*qxi**2 - 2*qzi**2)*(-2*qwij*qxij + 2*qyij*qzij),
            (1 - 2*qxi**2 - 2*qyi**2)*(1 - 2*qxij**2 - 2*qyij**2) + 
            (-2*qwi*qyi + 2*qxi*qzi)*(2*qwij*qyij + 2*qxij*qzij) + 
            (2*qwi*qxi + 2*qyi*qzi)*(-2*qwij*qxij + 2*qyij*qzij),0,0,0,0],
        [0,0,0,qwi*qwij - qxi*qxij - qyi*qyij - qzi*qzij,
            -(qxij*qyi) + qxi*qyij + qwij*qzi + qwi*qzij,
            -(qwij*qyi) - qwi*qyij - qxij*qzi + qxi*qzij,
            -(qwij*qxi) - qwi*qxij + qyij*qzi - qyi*qzij],
        [0,0,0,qxij*qyi - qxi*qyij - qwij*qzi - qwi*qzij,
            qwi*qwij - qxi*qxij - qyi*qyij - qzi*qzij,
            qwij*qxi + qwi*qxij - qyij*qzi + qyi*qzij,
            -(qwij*qyi) - qwi*qyij - qxij*qzi + qxi*qzij],
        [0,0,0,qwij*qyi + qwi*qyij + qxij*qzi - qxi*qzij,
            -(qwij*qxi) - qwi*qxij + qyij*qzi - qyi*qzij,
            qwi*qwij - qxi*qxij - qyi*qyij - qzi*qzij,
            qxij*qyi - qxi*qyij - qwij*qzi - qwi*qzij]]
    return np.asarray(res, dtype=np.float64)
