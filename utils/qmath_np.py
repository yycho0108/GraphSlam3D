import numpy as np
from tf import transformations as tx

eps = np.finfo(float).eps

def qinv(q):
    qx,qy,qz,qw = q
    return np.asarray([-qx,-qy,-qz,qw], dtype=np.float64)

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

mode = 'v1'

if mode == 'v1':
    #V1 : T(q) = q[:3]
    def T(q):
        return q[:3]

    def Tinv(q):
        x,y,z = q
        try:
            d = 1.0 - x**2 - y**2 - z**2
            d = max(0.0, d) # TODO : protection, but valid?
            w = np.sqrt(d)
        except Exception as e:
            print e
            print x, y, z
            print x**2 + y**2 + z**2
            raise e
        return np.asarray([x,y,z,w], dtype=np.float64)

    def dTdX(x):
        return np.eye(3,4, dtype=np.float64)

    def M(p, q):
        """
        Manifold projection  {d(x (+) dx) / dx}| dx==0
        """
        res = np.zeros((7,6), dtype=np.float64)
        res[:3,:3] = np.eye(3)
        res[3:,3:] = qr2Q(q).dot(np.eye(4,3))
        return res
else:
    # V2 : T(q) = log(q)
    def sinc(x):
        if np.abs(x) < eps:
            return 1.0
        else:
            return np.sin(x) / x
    def T(q):
        """ Quaternion Log """
        va, ra = q[:3], q[-1]
        ra = np.clip(ra, -1.0, 1.0)
        n = np.linalg.norm(va)
    
        if n < eps:
            # zero-vector
            return va * 0.0
        try:
            res = (va/n) * (np.arccos(ra))
        except Exception as e:
            print ra
            print e
            raise e
        return res
    
    def Tinv(qv):
        """ Rotation-Vector Exponential """
        ac = np.linalg.norm(qv, axis=-1) # == ac
        ra = np.cos(ac)
        va = sinc(ac) * qv # handles ac==0
        q = np.concatenate([va, [ra]], axis=-1)
        return q
    
    def dTdX(x):
        x = x / np.linalg.norm(x)
        qx,qy,qz,qw = x
    
        #qw = np.clip(qw, -1.0, 1.0)
        h = np.arccos(qw)
        n2 = (qx**2 + qy**2 + qz**2)
        s  = np.sqrt(1 - qw**2)
    
        if n2 < eps or s < eps:
            return np.zeros((3,4))
    
        res = [[h*(n2-qx**2), -h*qx*qy, -h*qx*qz, -n2*qx/s],
                [-h*qx*qy, h*(n2-qy**2), -h*qy*qz, -n2*qy/s],
                [-h*qx*qz, -h*qy*qz, h*(n2-qz**2), -n2*qz/s]]
        res = np.divide(res, n2**1.5)
    
        return np.asarray(res, dtype=np.float64)
    
    def M(p,q):
        """
        Manifold projection  {d(x (+) dx) / dx}| dx==0
        """
        res = np.zeros((7,6), dtype=np.float64)
        res[:3,:3] = np.eye(3)
        res[3:,3:] = qr2Q(q).dot(np.eye(4,3))
        return res

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



def dRTdq(q):
    """ d(R(q).T)/d(q) """
    x,y,z,w = q
    x2,y2,z2,w2 = [2*e for e in q]
    res = [[[0,-y2,-z2,0],[y,x,w,z],[z,-w,x,-y]],
            [[y,x,-w,-z],[-x2,0,-z2,0],[w,z,y,x]],
            [[z,w,x,y],[-w,z,y,-x],[-x2,-y2,0,0]]]
    return np.multiply(res, 2)

def dqidq(q):
    """ d q^{-1})/d(q) """
    return np.diag([-1,-1,-1,1])

def ql2Q(q):
    """
    4x4 Matrix representing left-multiplied quaternion, such that
    ql2Q(ql).qr == ql.qr
    """
    x,y,z,w = q
    res = [[w,-z,y,x],[z,w,-x,y],[-y,x,w,z],[-x,-y,-z,w]]
    return np.asarray(res, dtype=np.float64)

def qr2Q(q):
    """
    4x4 Matrix representing right-multiplied quaternion, such that
    qr2Q(qr).ql == ql.qr
    """
    x,y,z,w = q
    res = [[w,z,-y,x],[-z,w,x,y],[y,-x,w,z],[-x,-y,-z,w]]
    return np.asarray(res, dtype=np.float64)


def Aij_Bij_eij(p0, p1, dp, q0, q1, dq):
    RqiT = q2R(q0).T
    A01 = np.einsum('ijk,j->ik', dRTdq(q0), p0-p1)
    # note p0-p1 due to sign inversion

    dqi = qinv(dq)
    q0i = qinv(q0)

    # compute error
    ep = dp - RqiT.dot(p1 - p0)
    eq = qmul(dqi, qmul(q0i, q1))
    dTde = dTdX(eq)
    dQi = ql2Q(dqi)

    A11 = dTde.dot(dQi.dot(qr2Q(q1).dot(dqidq(q0))))
    B11 = dTde.dot(dQi.dot(ql2Q(q0i)))

    # construct Aij
    Aij = np.zeros((6,7), dtype=np.float64)
    Aij[:3,:3] = RqiT
    Aij[:3,3:] = A01
    Aij[3:,3:] = A11

    Mi = M(p0, q0)
    Aij = Aij.dot(Mi)


    # construct Bij
    Bij = np.zeros((6,7), dtype=np.float64)
    Bij[:3,:3] = -RqiT
    Bij[3:,3:] = B11

    Mj = M(p1, q1)
    Bij = Bij.dot(Mj)

    eij = np.concatenate([ep, T(eq)], axis=0)
    eij = np.expand_dims(eij, axis=1) # --> (6x1)

    return Aij, Bij, eij

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
        [np.sin(q), np.cos(q)]], dtype=np.float64)
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
    #np.random.seed(1358)
    p0 = rt()
    p1 = rt()

    q0 = rq()
    q1 = rq()

    eq = rq(s=1.0) 
    ep = rt(s=1.0)

    print p0
    print p1

    print q0
    print q1

    p01_z = qxv(q0, p1-p0) - ep
    q01_z = qmul(eq, qmul(qinv(q0), q1))

    print p01_z
    print q01_z

    print eij(p0,p1,p01_z, q0,q1, q01_z)

    #print q0
    #print q1

    #print 'q0', q0
    #print 'q0-bf', Tinv(T(q0))

    #p01_z = qxv(q0, p1-p0)
    #q01_z = qmul(qinv(q0), q1)

    #dp = qxv(q0, p1-p0) - p01_z
    #dq = qmul(qmul(qinv(q0),q1),qinv(q01_z))
    #print dp, dq

    #A = Aij(p0,p1,dp, q0,q1,dq)
    #B = Bij(p0,p1,dp, q0,q1,dq)

if __name__ == "__main__":
    main()
