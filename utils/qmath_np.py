import numpy as np
from tf import transformations as tx

from qmath_np_ex import Aij as Aij_alt
from qmath_np_ex import Bij as Bij_alt

eps = np.finfo(float).eps

mode = 'abs'

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
    return np.asarray(R, dtype=np.float64)

def qxv(q, v):
    return q2R(q).dot(v)

def dRqTpdq(q, p):
    # == d(R(q).T.p) / dq 
    qx,qy,qz,qw = q
    x,y,z = p
    J = [[2*qy*y + 2*qz*z, -4*qy*x + 2*qx*y - 2*qw*z, -4*qz*x + 2*qw*y + 2*qx*z, 2*qz*y - 2*qy*z],
         [2*qy*x - 4*qx*y + 2*qw*z, 2*qx*x + 2*qz*z, -2*qw*x - 4*qz*y + 2*qy*z, -2*qz*x + 2*qx*z],
         [2*qz*x - 2*qw*y - 4*qx*z, 2*qw*x + 2*qz*y - 4*qy*z, 2*qx*x + 2*qy*y, 2*qy*x - 2*qx*y]]
    return np.asarray(J, dtype=np.float64)

def dQdq0(q1,q2,q3):
    # q2 = qi
    # q3 = qj
    # q4 = q_ij
    # == d(q0^{-1}.q1.q01z^{-1}) / d(q0)
    qx1, qy1, qz1, qw1 = q1
    qx2, qy2, qz2, qw2 = q2
    qx3, qy3, qz3, qw3 = q3

    dqq = [[-qw2*qw3-qx2*qx3-qy2*qy3-qz2*qz3,-qx3*qy2+qx2*qy3-qw3*qz2+qw2*qz3,qw3*qy2-qw2*qy3-qx3*qz2+qx2*qz3,qw3*qx2-qw2*qx3+qy3*qz2-qy2*qz3],[qx3*qy2-qx2*qy3+qw3*qz2-qw2*qz3,-qw2*qw3-qx2*qx3-qy2*qy3-qz2*qz3,-qw3*qx2+qw2*qx3-qy3*qz2+qy2*qz3,qw3*qy2-qw2*qy3-qx3*qz2+qx2*qz3],[-qw3*qy2+qw2*qy3+qx3*qz2-qx2*qz3,qw3*qx2-qw2*qx3+qy3*qz2-qy2*qz3,-qw2*qw3-qx2*qx3-qy2*qy3-qz2*qz3,qx3*qy2-qx2*qy3+qw3*qz2-qw2*qz3],[qw3*qx2-qw2*qx3+qy3*qz2-qy2*qz3,qw3*qy2-qw2*qy3-qx3*qz2+qx2*qz3,qx3*qy2-qx2*qy3+qw3*qz2-qw2*qz3,qw2*qw3+qx2*qx3+qy2*qy3+qz2*qz3]]
    return np.asarray(dqq, dtype=np.float64)

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
    return np.asarray(dqq, dtype=np.float64)

#V1 : T(q) = q[:3]
def T(q):
    return q[:3]

def Tinv(q):
    x,y,z = q
    try:
        w = np.sqrt(1.0  - x**2 - y**2 - z**2)
    except Exception as e:
        print e
        print x, y, z
        print x**2 + y**2 + z**2
        raise e
    return np.asarray([x,y,z,w], dtype=np.float64)

def dTdX(x):
    return np.eye(3,4, dtype=np.float64)

# V2 : T(q) = log(q)
#def sinc(x):
#    if np.abs(x) < eps:
#        return 1.0
#    else:
#        return np.sin(x) / x
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
#def Tinv(qv):
#    """ Rotation-Vector Exponential """
#    ac = np.linalg.norm(qv, axis=-1) # == ac
#    ra = np.cos(ac)
#    va = sinc(ac) * qv # handles ac==0
#    q = np.concatenate([va, [ra]], axis=-1)
#    return q
#
#def dTdX(x):
#    qx,qy,qz,qw = x
#    #print 'qw', qw
#    qw = np.clip(qw, -1.0, 1.0)
#
#    h  = np.arccos(qw)
#    nv = np.sqrt(qx**2 + qy**2 + qz**2)
#    nv_1_5 = nv ** 1.5
#    s  = np.sqrt(1 - qw**2)
#
#    if nv < eps or s < eps:
#        return np.zeros((3,4))
#    #print 'h', h
#    #print 'nv', nv
#    #print 's', s
#    #print 'x', x
#
#    res = [
#            [-((qx**2*h)/(nv_1_5)) + 
#                h/nv,
#                -((qx*qy*h)/(nv_1_5)),
#                -((qx*qz*h)/(nv_1_5)),
#                -(qx/(s*nv))],
#            [-((qx*qy*h)/(nv_1_5)),
#                -((qy**2*h)/(nv_1_5)) + 
#                h/nv,
#                -((qy*qz*h)/(nv_1_5)),
#                -(qy/(s*nv))],
#            [-((qx*qz*h)/(nv_1_5)),
#                -((qy*qz*h)/(nv_1_5)),
#                -((qz**2*h)/(nv_1_5)) + 
#                h/nv,
#                -(qz/(s*nv))]]
#    return np.asarray(res)

# 
# def dTdX(x):
#     x = np.divide(x, np.linalg.norm(x))
#     qxi,qyi,qzi,qwi = x
# 
#     # prevent minor numerical issues
#     # qwi = np.clip(qwi, -1.0, 1.0)
# 
#     h = np.arccos(qwi)
# 
#     k = (1 - qwi**2)
# 
#     qvn = np.sqrt(qxi**2 + qyi**2 + qzi**2)
# 
#     if qvn < eps:
#         # TODO : valid?
#         return np.zeros((3,4), dtype=np.float64)
#     else:
#         d = k * qvn 
#         qvn1_5 = qvn**1.5
#         res = [
#                 [((qyi**2 + qzi**2)*h)/qvn1_5,
#                     ((qxi*qyi*h)/qvn1_5),
#                     ((qxi*qzi*h)/qvn1_5),
#                     (qxi/d)],
#                 [((qxi*qyi*h)/qvn1_5),
#                     ((qxi**2 + qzi**2)*h)/qvn1_5,
#                     ((qyi*qzi*h)/qvn1_5),
#                     (qyi/d)],
#                 [((qxi*qzi*h)/qvn1_5),
#                     ((qyi*qzi*h)/qvn1_5),
#                     ((qxi**2 + qyi**2)*h)/qvn1_5,
#                     (qzi/d)]]
#         return np.asarray(res)

def xadd_rel(x, dx, T=True):
    """ apply dx to x in relative frames """
    p, q = x2pq(x)
    dp, dq = x2pq(dx)
    dq = Tinv(dq) if T else dq
    p_n = p + qxv(q, dp)
    q_n = qmul(q, dq)
    return pq2x(p_n, q_n)

def xadd_abs(x, dx, T=True):
    """ apply dx to x in absolute frames """
    p, q = x2pq(x)
    dp, dq = x2pq(dx)
    dq = Tinv(dq) if T else dq
    p_n = p + dp
    q_n = qmul(dq, q)
    return pq2x(p_n, q_n)

## x+dx, v1 : relative addition
def dqnddq_rel(q):
    x,y,z,w = q
    res = [[w,-z,y],[z,w,-x],[-y,x,w],[-x,-y,-z]]
    return np.asarray(res, dtype=np.float64)

def M_rel(p, q):
    M = np.zeros((7,6), dtype=np.float64)
    M[:3,:3] = q2R(q)
    M[3:,3:] = dqnddq_rel(q)
    return M

# x+dx, v2 : absolute addition
def dqnddq_abs(q):
    x,y,z,w = q
    res = [[w,z,-y],[-z,w,x],[y,-x,w],[-x,-y,-z]]
    return np.asarray(res, dtype=np.float64)

def M_abs(p, q):
    M = np.zeros((7,6), dtype=np.float64)
    M[:3,:3] = np.eye(3)
    M[3:,3:] = dqnddq_abs(q)
    return M

if mode == 'abs':
    M = M_abs
    xadd = xadd_abs
else:
    M = M_rel
    xadd = xadd_rel

#def Aij(
#        p0, p1, dp,
#        q0, q1, dq,
#        ):
#    # == d(eij) / d(xi)
#    A = np.zeros((6,7), dtype=np.float64)
#    A[:3,:3] = -q2R(q0).T
#    A[:3,3:] = dRqTpdq(q0, p0)
#    A[3:,3:] = dTdX(dq).dot(dQdq0(q0, q1, dq))
#
#    Mi = M(p0, q0)
#    A = A.dot(Mi)
#    return A

def dRqidq(q, p1, p0):
    x,y,z,w = q
    x2,y2,z2,w2 = x*2, y*2, z*2, w*2
    x4,y4,z4,w4 = x*4, y*4, z*4, w*4
    res = [[[0,-y4,-z4,0],[y2,x2,w2,z2],[z2,-w2,x2,-y2]],
           [[y2,x2,-w2,-z2],[-x4,0,-z4,0],[w2,z2,y2,x2]],
           [[z2,w2,x2,y2],[-w2,z2,y2,-x2],[-x4,-y4,0,0]]]
    res = np.asarray(res)
    # == (3,3,4)
    dp = p1 - p0
    return np.einsum('ijk,j->ik', res, dp)

def dqq_l(q):
    # == d(q0.q1)/d(q1)
    x,y,z,w = q

    res = [[w,-z,y,x],
           [z,w,-x,y],
           [-y,x,w,z],
           [-x,-y,-z,w]]
    return np.asarray(res)

def dqiq_r(q):
    # == d(q0^{-1}.q1)/d(q0)
    x,y,z,w = q
    res = [[w,z,-y,-x],
            [-z,w,x,-y],
            [y,-x,w,-z],
            [-x,-y,-z,-w]]
    return np.asarray(res)

def dqedq1(q2, qe):
    qx2,qy2,qz2,qw2 = q2
    qxe,qye,qze,qwe = qe

    res = [
            [-(qw2*qwe) - qx2*qxe + qy2*qye + qz2*qze,
                -(qxe*qy2) - qx2*qye - qwe*qz2 - qw2*qze,qwe*qy2 + qw2*qye - qxe*qz2 - qx2*qze,
                qwe*qx2 - qw2*qxe - qye*qz2 + qy2*qze],
            [-(qxe*qy2) - qx2*qye + qwe*qz2 + qw2*qze,
                -(qw2*qwe) + qx2*qxe - qy2*qye + qz2*qze,
                -(qwe*qx2) - qw2*qxe - qye*qz2 - qy2*qze,qwe*qy2 - qw2*qye + qxe*qz2 - qx2*qze],
            [-(qwe*qy2) - qw2*qye - qxe*qz2 - qx2*qze,
                qwe*qx2 + qw2*qxe - qye*qz2 - qy2*qze,-(qw2*qwe) + qx2*qxe + qy2*qye - qz2*qze,
                -(qxe*qy2) + qx2*qye + qwe*qz2 - qw2*qze],
            [qwe*qx2 - qw2*qxe + qye*qz2 - qy2*qze,qwe*qy2 - qw2*qye - qxe*qz2 + qx2*qze,
                qxe*qy2 - qx2*qye + qwe*qz2 - qw2*qze,
                qw2*qwe + qx2*qxe + qy2*qye + qz2*qze]]
    return np.asarray(res)

def dqedq2(q1, qe):
    qx1,qy1,qz1,qw1 = q1
    qxe,qye,qze,qwe = qe

    res = [[qw1*qwe - qx1*qxe - qy1*qye - qz1*qze, -(qxe*qy1) + qx1*qye + qwe*qz1 + qw1*qze,
        -(qwe*qy1) - qw1*qye - qxe*qz1 + qx1*qze, -(qwe*qx1) - qw1*qxe + qye*qz1 - qy1*qze],
        [qxe*qy1 - qx1*qye - qwe*qz1 - qw1*qze,qw1*qwe - qx1*qxe - qy1*qye - qz1*qze,
            qwe*qx1 + qw1*qxe - qye*qz1 + qy1*qze,-(qwe*qy1) - qw1*qye - qxe*qz1 + qx1*qze],
        [qwe*qy1 + qw1*qye + qxe*qz1 - qx1*qze,
            -(qwe*qx1) - qw1*qxe + qye*qz1 - qy1*qze,qw1*qwe - qx1*qxe - qy1*qye - qz1*qze,
            qxe*qy1 - qx1*qye - qwe*qz1 - qw1*qze],
        [qwe*qx1 + qw1*qxe - qye*qz1 + qy1*qze,qwe*qy1 + qw1*qye + qxe*qz1 - qx1*qze,
            -(qxe*qy1) + qx1*qye + qwe*qz1 + qw1*qze,qw1*qwe - qx1*qxe - qy1*qye - qz1*qze]]
    return np.asarray(res)

def Aij(p0,p1,dp, q0,q1,dq):
    # == d(eij) / d(xi)

    A = np.zeros((6,7), dtype=np.float64)

    R01 = q2R(dq)
    A[:3,:3] = -R01.T.dot(q2R(q0).T)
    A[:3,3:] = R01.T.dot(dRqidq(q0, p1, p0))

    eq = qmul(qinv(dq), qmul(qinv(q0),q1))

    #Q01 = dqq_l(qinv(dq))
    #A[3:,3:] = dTdX(eq).dot(Q01.dot(dqiq_r(q1)))
    A[3:,3:] = dTdX(eq).dot(dqedq1(q1,dq)) # this is correct

    #vs1 = Q01.dot(dqiq_r(q1))
    #vs2 = dqedq1(q1,dq)
    #print '=='
    #print vs1
    #print vs2
    #print vs1 - vs2

    #A2 = Aij_alt(p0,p1,dp,q0,q1,dq)
    #print np.mean(np.abs(A - A2)[3:,3:])

    Mi = M(p0, q0)
    A = A.dot(Mi)
    return A

#def Bij(
#        p0, p1, dp,
#        q0, q1, dq,
#        ):
#    # == d(eij) / d(xj)
#    B = np.zeros((6,7), dtype=np.float64)
#    B[:3,:3] = q2R(q0).T
#    B[3:,3:] = dTdX(dq).dot(dQdq1(q0, q1, dq))
#
#    Mj = M(p1, q1)
#    B = B.dot(Mj)
#    return B

def Bij(
        p0, p1, dp,
        q0, q1, dq,
        ):
    # == d(eij) / d(xj)
    R0 = q2R(q0)
    R01 = q2R(dq)

    B = np.zeros((6,7), dtype=np.float64)
    B[:3,:3] = R01.T.dot(R0.T)

    Q01 = dqq_l(qinv(dq))
    Q12 = dqq_l(qinv(q0))
    eq = qmul(qinv(dq), qmul(qinv(q0),q1))

    # below two are pretty much equivalent
    #B[3:,3:] = dTdX(eq).dot(Q01.dot(Q12))
    B[3:,3:] = dTdX(eq).dot(dqedq2(q0, dq))

    #B2 = Bij_alt(p0,p1,dp,q0,q1,dq)
    #print np.sum(np.abs(B-B2)[3:,3:])
    #print B - B2

    Mj = M(p1, q1)
    B = B.dot(Mj)
    return B

#def eij(
#        p0, p1, dp,
#        q0, q1, dq,
#        ):
#    # ep = q0^{-1}.(p1-p0) - dp
#    # eq = T(q0^{-1}.q1.dp^{-1})
#
#    # estimated dpe
#    dp_e, dq_e = xrel(p0, q0, p1, q1)
#
#    err_p = dp_e - dp
#    err_q = qmul(dq_e, qinv(dq))
#    err_q = T(err_q)
#
#    #q0i = qinv(q0)
#    #ep = qxv(q0i, (p1 - p0)) - dp
#    #eq = T(qmul(q0i, qmul(q1, qinv(dq))))
#    res = np.concatenate([err_p,err_q], axis=-1)
#    return np.expand_dims(res, axis=-1) # (6,1)

def eij(p0, p1, dp, q0, q1, dq):
    dqi = qinv(dq)
    ep = qxv(dqi, qxv(qinv(q0), p1-p0) - dp)
    eq = T(qmul(dqi, qmul(qinv(q0), q1)))
    res = np.concatenate([ep, eq], axis=-1)
    return np.expand_dims(res, axis=-1)

def x2pq(x):
    p = x[:3]
    q = x[3:]
    return p, q

def pq2x(p, q):
    return np.concatenate([p,q], axis=-1)


def rt(s=1.0):
    """ random position """
    return np.random.uniform(-s, s, size=3)

def rq(s=np.pi):
    """ random quaternion """
    ax = rt()
    ax /= np.linalg.norm(ax)
    h = np.random.uniform(-s, s)
    return tx.quaternion_about_axis(h,ax)

def xrel(p0, q0, p1, q1):
    """ convert absolute frames (p1,q1) to relative frames """
    q0i = qinv(q0)
    pn = qxv(q0i, p1 - p0)
    qn = qmul(q0i, q1)
    return pn, qn

def xabs(p0, q0, p1, q1):
    """ convert relative frames (p1,q1) to absolute frames """
    pn = p0 + qxv(q0, p1)
    qn = qmul(q0, q1)
    return pn, qn

# 2D versions ...

def hnorm2(h):
    return ((h + np.pi) % (2*np.pi)) - np.pi

def q2R2(q):
    q = q[0]
    R = np.asarray([
        [np.cos(q), -np.sin(q)],
        [np.sin(q), np.cos(q)]])
    return R

def rt2(s=1.0):
    return np.random.uniform(-s, s, size=2)

def rq2(s=np.pi):
    return np.random.uniform(-s, s, size=1)

def xrel2(p0, q0, p1, q1):
    dp = p1 - p0
    dpr = q2R2(-q0).dot(dp)
    dqr = hnorm2(q1 - q0)
    return dpr, dqr
    #xv = [np.cos(q0), np.sin(q0)]
    #yv = [np.cos(q0+np.pi/2), np.sin(q0+np.pi/2)]
    #dxr = np.dot(xv, dp)
    #dyr = np.dot(yv, dp)
    ## dpr = R(q0).T.dp
    #dq = q1 - q0
    #dq = np.arctan2(np.sin(dq), np.cos(dq))
    #return [dxr,dyr], [dq]

def dR2Tdq2(q):
    c, s = np.cos(q), np.sin(q)
    return np.reshape([-s,c,-c,-s], (2,2))

def Aij2(p0, p1, dp, q0, q1, dq):
    Aij = np.zeros((3,3), dtype=np.float64)
    R0 = q2R2(q0)
    R1 = q2R2(q1)
    R01 = q2R2(dq)

    dRi = dR2Tdq2(q0)

    #zt = np.zeros((3,3))
    #zt[:2,:2] = R01
    #zt[:2,2]  = dp
    #zt[2,2]   = 1.0
    #zti = np.linalg.pinv(zt)
    #zti[:2,2] = 0.0
    #print zti

    Aij[:2,:2] = -R01.T.dot(R0.T)
    Aij[:2,2] = R01.T.dot(dRi.dot(p1-p0))
    Aij[2,2] = -1

    #print '?'
    #print Aij

    #Aij *= 0.0

    #Aij[:2,:2] = -R0.T
    #Aij[:2,2]  = dRi.dot(p1-p0)
    #Aij[2,2] = -1.0
    #Aij = zti.dot(Aij)

    #print '??'
    #print Aij
    return Aij

def Bij2(p0, p1, dp, q0, q1, dq):
    Bij = np.zeros((3,3), dtype=np.float64)
    R0 = q2R2(q0)
    R1 = q2R2(q1)
    R01 = q2R2(dq)

    #zt = np.zeros((3,3))
    #zt[:2,:2] = q2R2(dq)
    #zt[:2,2]  = dp
    #zt[2,2]  = 1.0

    #zti = np.linalg.pinv(zt)
    #zti[:2,2] = 0.0

    #Bij[:2,:2] = R0.T
    #Bij[2,2]   = 1.0
    #Bij = zti.dot(Bij)

    Bij[:2,:2] = R01.T.dot(R0.T)
    Bij[2,2] = 1

    return Bij

def eij2test(p0, p1, dp, q0, q1, dq):
    eij = np.zeros((3,1), dtype=np.float64)

    T01 = np.zeros((3,3))
    T0 = np.zeros((3,3))
    T1 = np.zeros((3,3))

    T0[:2,:2] = q2R2(q0)
    T0[:2,2] = p0
    T0[2,2] = 1

    T01[:2,:2] = q2R2(dq)
    T01[:2,2]  = dp
    T01[2,2] = 1

    T1[:2,:2] = q2R2(q1)
    T1[:2,2] = p1
    T1[2,2] = 1

    inv = np.linalg.pinv

    #print np.dot( inv(T01), inv(T0))
    #print inv(T01).dot(inv(T0))

    Terr = (inv(T01).dot(inv(T0))).dot(T1)
    #Terr = np.dot(np.dot(inv(T01), inv(T0)), T1)
    eij[:2, 0] = Terr[:2, 2]
    eij[2]     = np.arctan2(Terr[1,0], Terr[0,0])
    return eij

def eij2(p0, p1, dp, q0, q1, dq):
    eij = np.zeros((3,1), dtype=np.float64)
    R0 = q2R2(q0)
    R1 = q2R2(q1)
    R01 = q2R2(dq)

    eij[:2,0] = R01.T.dot(R0.T.dot(p1-p0)-dp)
    eij[2,0] = hnorm2(q1-q0-dq)

    #zt = np.zeros((3,3))
    #zt[:2,:2] = q2R2(dq)
    #zt[:2,2]  = dp
    #zt[2,2]  = 1.0
    #zti = np.linalg.pinv(zt)

    #fij = np.zeros((3,3))
    #fij[:2,:2] = R0.T.dot(R1)
    #fij[:2,2]  = R0.T.dot(p1-p0)
    #fij[2,2]   = 1.0

    #eijT = zti.dot(fij)
    #eij[:2,0] = eijT[:2,2]
    #eij[2,0]  = np.arctan2(eijT[1,0], eijT[0,0])
    #eij[2,0]  = hnorm2(eij[2,0])
    #print eij

    return eij

def x2pq2(x):
    return x[:2], x[2:]

def xadd_rel2(x, dx, T=True):
    dxa = q2R2(x[-1:]).dot(dx[:2])
    xn = x[:2] + dxa
    qn = hnorm2(x[-1:] + dx[-1:])
    xn = np.concatenate([xn,qn], axis=-1)
    return xn

def xadd_abs2(x, dx, T=True):
    xn = x+dx
    xn[2] = hnorm2(xn[2])
    return xn

xadd2 = xadd_abs2
def main():
    p0 = rt()
    p1 = rt()

    q0 = rq()
    q1 = rq()

    print 'q0', q0
    print 'q0-bf', Tinv(T(q0))

    p01_z = qxv(q0, p1-p0)
    q01_z = qmul(qinv(q0), q1)

    dp = qxv(q0, p1-p0) - p01_z
    dq = qmul(qmul(qinv(q0),q1),qinv(q01_z))
    print dp, dq

    A = Aij(p0,p1,dp, q0,q1,dq)
    B = Bij(p0,p1,dp, q0,q1,dq)

if __name__ == "__main__":
    main()
