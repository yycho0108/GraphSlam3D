import numpy as np
from tf import transformations as tx

def x2pq(x):
    return x[:3], x[3:]

def pq2x(p, q):
    return np.concatenate([p,q], axis=-1)

def qlog(q):
    """ Quaternion Log """
    va, ra = q[:3], q[-1]
    uv = va / np.linalg.norm(va)
    return uv * np.arccos(ra)

def qvexp(qv):
    """ Rotation-Vector Exponential """
    ac = np.linalg.norm(qv, axis=-1)
    ra = np.cos(ac)
    va = (np.sin(ac)/ac) * qv
    q = np.concatenate([va, [ra]], axis=-1)
    return q / np.linalg.norm(q)

def qinv(q):
    """ Quaternion Inverse (Conjugate, Unit) """
    # no normalization, assume unit
    #va, ra = q[:3], q[-1]
    #return np.concatenate([va, [-ra]], axis=-1)
    return tx.quaternion_conjugate(q)

def qmul(b,a):
    """ Quaternion Multiply """
    return tx.quaternion_multiply(b, a)

def qxv(q, v, norm=True):
    """
    Multiply vector by a quaternion.
    (Effectively, a rotation)

    q : xyzw
    v : xyz
    """
    if not norm:
        s = np.linalg.norm(v)
        v = v / s # make unit
    q_v = list(v) + [0]
    q_c = qinv(q)
    v = qmul(qmul(q, q_v),q_c)[:-1] # == q.v.q^{-1}
    if not norm:
        v = np.multiply(v,s)
    return v

def parametrize(x):
    p, q = x[:3], x[3:]
    q /= np.linalg.norm(q)
    qv = qlog(q)
    return np.concatenate([p,qv], axis=-1)

def unparametrize(x):
    p, qv = x[:3], x[3:]
    q = qvexp(qv)
    q /= np.linalg.norm(q)
    return np.concatenate([p,q], axis=-1)

def motion_compose(xi, xj):
    pi, qi = xi[:3], xi[3:]
    pj, qj = xj[:3], xj[3:]
    
    p = pi + qxv(qi,pj)
    q = qmul(qi, qj)
    return np.concatenate([p,q], axis=-1)

def xinv(x):
    # TODO : maybe?
    p, q = x[:3], x[3:]
    return np.concatenate([-p, qinv(q)], axis=-1)

def xadd(xi, dxi):
    motion_compose(xi, unparametrize(dxi))

def error(xi, xj, zij):
    pi, qi = x2pq(xi)
    pj, qj = x2pq(xj)
    pij_z, qij_z = x2pq(zij)

    est_p = qxv(qinv(qi), pj - pi)
    est_q = qmul(qinv(qi), qj)

    err_p = est_p - pij_z
    err_q = 2.0 * (qmul(est_q, qinv(qij_z)))[:3]
    e1 = np.concatenate((err_p, err_q), axis=-1)
    e2 = motion_compose(xinv(zij), motion_compose(xinv(xi), xj))
    print e1
    print e2[:6]
    e = e1
    return e

    #return parametrize(e)

def random_pose():
    p = np.random.normal(size=3)
    ax = np.random.uniform(size=3)
    ax /= np.linalg.norm(ax)
    h  = np.random.uniform(low=-np.pi, high=np.pi, size=1)
    q = tx.quaternion_about_axis(h, ax)
    return np.concatenate((p,q), axis=-1)


xi = random_pose()
xj = random_pose()

pi,qi = x2pq(xi)
pj,qj = x2pq(xj)

dp = qxv(qinv(qi), pj - pi)#qxv(qi,pj) - pi
dq = qmul(qinv(qi), qj)#qmul(qj, qinv(qi))
# ^^ above are "relative" measurements wrt. frame i

zij = pq2x(dp, dq)
#zij = random_pose()

#xi = np.concatenate(
#        ([0,0,0], [0,0,0,1]), axis=-1)
#xj = np.concatenate(
#        ([0,0,0], [0,0,0,1]), axis=-1)
#zij = np.concatenate(
#        ([0,0,0], [0,0,0,1]), axis=-1)
error(xi, xj, zij)
