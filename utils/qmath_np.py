import numpy as np
from tf import transformations as tx

eps = np.finfo(float).eps

def qinv(q):
    qx,qy,qz,qw = q
    return np.asarray([-qx,-qy,-qz,qw])

def qmul(q1, q0):
    return tx.quaternion_multiply(q1,q0)

def q2R(q):
    qx,qy,qz,qw = q
    qx2,qy2,qz2,qw2 = np.square(q)
    R = [[1 - 2*qy2 - 2*qz2,	2*qx*qy - 2*qz*qw,	2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,	1 - 2*qx2 - 2*qz2,	2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,	2*qy*qz + 2*qx*qw,	1 - 2*qx2 - 2*qy2]]
    return np.asarray(R, dtype=np.float32)

def qxv(q, v):
    return q2R(q).dot(v)

def dRqTpdq(q, p):
    # == d(R(q).T.p) / dq 
    qx,qy,qz,qw = q
    x,y,z = p
    J = [[2*qy*y + 2*qz*z, -4*qy*x + 2*qx*y - 2*qw*z, -4*qz*x + 2*qw*y + 2*qx*z, 2*qz*y - 2*qy*z],
         [2*qy*x - 4*qx*y + 2*qw*z, 2*qx*x + 2*qz*z, -2*qw*x - 4*qz*y + 2*qy*z, -2*qz*x + 2*qx*z],
         [2*qz*x - 2*qw*y - 4*qx*z, 2*qw*x + 2*qz*y - 4*qy*z, 2*qx*x + 2*qy*y, 2*qy*x - 2*qx*y]]
    return np.asarray(J, dtype=np.float32)

def dQdq0(q1,q2,q3):
    # q2 = qi
    # q3 = qj
    # q4 = q_ij
    # == d(q0^{-1}.q1.q01z^{-1}) / d(q0)
    qx1, qy1, qz1, qw1 = q1
    qx2, qy2, qz2, qw2 = q2
    qx3, qy3, qz3, qw3 = q3

    dqq = [[-qw2*qw3-qx2*qx3-qy2*qy3-qz2*qz3,-qx3*qy2+qx2*qy3-qw3*qz2+qw2*qz3,qw3*qy2-qw2*qy3-qx3*qz2+qx2*qz3,qw3*qx2-qw2*qx3+qy3*qz2-qy2*qz3],[qx3*qy2-qx2*qy3+qw3*qz2-qw2*qz3,-qw2*qw3-qx2*qx3-qy2*qy3-qz2*qz3,-qw3*qx2+qw2*qx3-qy3*qz2+qy2*qz3,qw3*qy2-qw2*qy3-qx3*qz2+qx2*qz3],[-qw3*qy2+qw2*qy3+qx3*qz2-qx2*qz3,qw3*qx2-qw2*qx3+qy3*qz2-qy2*qz3,-qw2*qw3-qx2*qx3-qy2*qy3-qz2*qz3,qx3*qy2-qx2*qy3+qw3*qz2-qw2*qz3],[qw3*qx2-qw2*qx3+qy3*qz2-qy2*qz3,qw3*qy2-qw2*qy3-qx3*qz2+qx2*qz3,qx3*qy2-qx2*qy3+qw3*qz2-qw2*qz3,qw2*qw3+qx2*qx3+qy2*qy3+qz2*qz3]]
    return np.asarray(dqq, dtype=np.float32)

def dQdq1(q1, q2, q3):
    # q2 = qi = q0
    # q3 = qj = q1
    # q4 = q_ij = q01
    # == d(q0^{-1}.q1.q01z^{-1}) / d(q1)
    
    qx1, qy1, qz1, qw1 = q1
    qx2, qy2, qz2, qw2 = q2
    qx3, qy3, qz3, qw3 = q3

    dqq = [[qw1*qw3 - qx1*qx3 + qy1*qy3 + qz1*qz3, -qx3*qy1 - qx1*qy3 + qw3*qz1 - qw1*qz3, -qw3*qy1 + qw1*qy3 - qx3*qz1 - qx1*qz3, -qw3*qx1 - qw1*qx3 - qy3*qz1 + qy1*qz3],
            [-qx3*qy1 - qx1*qy3 - qw3*qz1 + qw1*qz3, qw1*qw3 + qx1*qx3 - qy1*qy3 + qz1*qz3, qw3*qx1 - qw1*qx3 - qy3*qz1 - qy1*qz3, -qw3*qy1 - qw1*qy3 + qx3*qz1 - qx1*qz3],
            [qw3*qy1 - qw1*qy3 - qx3*qz1 - qx1*qz3, -qw3*qx1 + qw1*qx3 - qy3*qz1 - qy1*qz3, qw1*qw3 + qx1*qx3 + qy1*qy3 - qz1*qz3, -qx3*qy1 + qx1*qy3 - qw3*qz1 - qw1*qz3],
            [qw3*qx1 + qw1*qx3 - qy3*qz1 + qy1*qz3, qw3*qy1 + qw1*qy3 + qx3*qz1 - qx1*qz3, -qx3*qy1 + qx1*qy3 + qw3*qz1 + qw1*qz3, qw1*qw3 - qx1*qx3 - qy1*qy3 - qz1*qz3]]
    return np.asarray(dqq, dtype=np.float32)

# V1 : T(q) = q[:3]
def T(q):
    return q[:3]

def Tinv(q):
    x,y,z = q
    w = np.sqrt(1.0  - x**2 - y**2 - z**2)
    return np.asarray([x,y,z,w], dtype=np.float32)

def dTdX(x):
    return np.eye(3,4, dtype=np.float32)

# V2 : T(q) = log(q)
#def T(q):
#    """ Quaternion Log """
#    va, ra = q[:3], q[-1]
#    ra = np.clip(ra, -1.0, 1.0)
#    n = np.linalg.norm(va)
#
#    if n < eps:
#        # zero-vector
#        return va * 0.0
#    try:
#        res = (va/n) * (np.arccos(ra))
#    except Exception as e:
#        print ra
#        print e
#        raise e
#    return res
#
#def sinc(x):
#    if np.abs(x) < eps:
#        return 1.0
#    else:
#        return np.sin(x) / x
#
#def Tinv(qv):
#    """ Rotation-Vector Exponential """
#    ac = np.linalg.norm(qv, axis=-1) # == ac
#    ra = np.cos(ac)
#    va = sinc(ac) * qv # handles ac==0
#    q = np.concatenate([va, [ra]], axis=-1)
#    return q
#
#def dTdX(x):
#    qxi,qyi,qzi,qwi = x
#
#    qwi = np.clip(qwi, -1.0, 1.0) # prevent minor numerical issues
#
#    h = np.arccos(qwi)
#
#    k = (1 - qwi**2)
#
#    qvn = np.sqrt(qxi**2 + qyi**2 + qzi**2)
#
#    if qvn < eps:
#        # TODO : valid?
#        return np.zeros((3,4), dtype=np.float32)
#    else:
#        d = k * qvn 
#        qvn1_5 = qvn**1.5
#        res = [
#                [((qyi**2 + qzi**2)*h)/qvn1_5,
#                    ((qxi*qyi*h)/qvn1_5),
#                    ((qxi*qzi*h)/qvn1_5),
#                    (qxi/d)],
#                [((qxi*qyi*h)/qvn1_5),
#                    ((qxi**2 + qzi**2)*h)/qvn1_5,
#                    ((qyi*qzi*h)/qvn1_5),
#                    (qyi/d)],
#                [((qxi*qzi*h)/qvn1_5),
#                    ((qyi*qzi*h)/qvn1_5),
#                    ((qxi**2 + qyi**2)*h)/qvn1_5,
#                    (qzi/d)]]
#        return np.asarray(res)

def dqnddq(q):
    x,y,z,w = q
    res = [[w,-z,y],[z,w,-x],[-y,x,w],[-x,-y,-z]]
    return np.asarray(res, dtype=np.float32)

def M(p, q):
    # TODO : maybe wrong right now
    M = np.zeros((7,6), dtype=np.float32)
    M[:3,:3] = q2R(q)
    M[3:,3:] = dqnddq(q)
    return M

def Aij(
        p0, p1, dp,
        q0, q1, dq,
        ):
    # == d(eij) / d(xi)
    A = np.zeros((6,7), dtype=np.float32)
    A[:3,:3] = -q2R(q0).T
    A[:3,3:] = dRqTpdq(q0, p0)
    A[3:,3:] = dTdX(dq).dot(dQdq0(q0, q1, dq))

    Mi = M(p0, q0)
    A = A.dot(Mi)
    return A

def Bij(
        p0, p1, dp,
        q0, q1, dq,
        ):
    # == d(eij) / d(xj)
    B = np.zeros((6,7), dtype=np.float32)
    B[:3,:3] = q2R(q0).T
    B[3:,3:] = dTdX(dq).dot(dQdq1(q0, q1, dq))

    Mj = M(p1, q1)
    B = B.dot(Mj)
    return B

def eij(
        p0, p1, dp,
        q0, q1, dq,
        ):
    # ep = q0^{-1}.(p1-p0) - dp
    # eq = T(q0^{-1}.q1.dp^{-1})

    ep = qxv(qinv(q0), (p1 - p0)) - dp
    eq = T(qmul(qinv(q0), qmul(q1, qinv(dq))))
    res = np.concatenate([ep,eq], axis=-1)
    return np.expand_dims(res, axis=-1) # (6,1)

def x2pq(x):
    p = x[:3]
    q = x[3:]
    return p, q

def pq2x(p, q):
    return np.concatenate([p,q], axis=-1)

def xadd(x, dx, T=True):
    p, q = x2pq(x)
    dp, dq = x2pq(dx)
    dq = Tinv(dq) if T else dq
    p_n = p + q2R(q).dot(dp)
    q_n = qmul(q, dq)
    return pq2x(p_n, q_n)

def rq():
    """ random quaternion """
    ax = np.random.uniform(-1.0,1.0,size=3)
    ax /= np.linalg.norm(ax)
    h = np.random.uniform(-np.pi,np.pi)
    return tx.quaternion_about_axis(h,ax)

def rt():
    """ random position """
    return np.random.uniform(-1.0, 1.0, size=3)

def xrel(p0, q0, p1, q1):
    """ convert absolute frames (p1,q1) to relative frames """
    pn = qxv(qinv(q0), p1 - p0)
    qn = qmul(qinv(q0), q1)
    return pn, qn

def xabs(p0, q0, p1, q1):
    """ convert relative frames (p1,q1) to absolute frames """
    pn = p0 + qxv(q0, p1)
    qn = qmul(q0, q1)
    return pn, qn

def main():
    p0 = rt()
    p1 = rt()

    q0 = rq()
    q1 = rq()

    print 'q0', q0
    print 'q0-bf', Tinv(T(q0))

    p01_z = q2R(q0).T.dot(p1-p0)
    q01_z = qmul(qinv(q0), q1)

    dp = q2R(q0).T.dot(p1-p0) - p01_z
    dq = qmul(qmul(qinv(q0),q1),qinv(q01_z))
    print dp, dq

    A = Aij(p0,p1,dp, q0,q1,dq)
    B = Bij(p0,p1,dp, q0,q1,dq)

if __name__ == "__main__":
    main()
