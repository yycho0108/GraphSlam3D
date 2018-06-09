import numpy as np
from utils import qmath_np

qinv = qmath_np.qinv
qmul = qmath_np.qmul
qxv  = qmath_np.qxv

from tf import transformations as tx

eps = np.finfo(float).eps

def random_quaternion(hmin=-np.pi, hmax=np.pi):
    ax = np.random.uniform(low=-1.0, high=1.0, size=3)
    h  = np.random.uniform(low=hmin, high=hmax, size=1)
    q = tx.quaternion_about_axis(h, ax)
    return q

def random_pose(xmin=-50.0, xmax=50.0):
    x = np.random.uniform(low=xmin, high=xmax, size=3)
    ax = np.random.uniform(low=-1.0, high=1.0, size=3)
    h  = np.random.uniform(low=-np.pi, high=np.pi, size=1)
    q = tx.quaternion_about_axis(h, ax)
    return x, q

def cat(a,b):
    return np.concatenate([a,b], axis=-1)

#def qinv(q):
#    """ Quaternion Inverse (Conjugate, Unit) """
#    return tx.quaternion_conjugate(q)
#
#def qmul(b,a):
#    return tx.quaternion_multiply(b, a)
#
#def qxv(q, v, norm=True):
#    s = np.linalg.norm(v)
#    if s < eps:
#        return 0 * v
#    v = v / s # make unit
#    q_v = np.concatenate((v,[0]), axis=-1)
#    q_c = qinv(q)
#    v = qmul(qmul(q, q_v),q_c)[:-1] # == q.v.q^{-1}
#    v = np.multiply(v,s)
#    return v

def gen_data(n_t, n_l,
        dx_p = 1.0,
        dx_q = 1.0,
        dz_p = 1.0,
        dz_q = 1.0
        ):    
    n_s = 6 # state size : xyz, qlog(q)

    graph = []

    # initial pose, (x0, q0)
    p0, q0 = random_pose()
    p0 *= 0.0 # ??
    x0  = cat(p0, q0)
    graph.append([0, 0, x0])

    zs = []
    for zi in range(n_l):
        zp, zq = random_pose()
        zs.append([zp, zq])

    p, q = p0.copy(), q0.copy()
    print 'initial'
    print p, q


    hmax = np.deg2rad(10)
    dwmax = np.deg2rad(1.0)
    zqmax = np.deg2rad(1.0)

    dq = random_quaternion(hmin=-hmax, hmax=hmax)

    v = 1.0

    obs = []

    # SPECIAL : add initial pose
    obs.append([0, 0, cat(p0, q0)])

    for i in range(1, n_t):
        # measure motion x0->x1
        ddq = random_quaternion(hmin=-dwmax, hmax=dwmax)
        dq = qmul(ddq, dq)

        # position; assume forward (x-axis) motion
        uv = qxv(q, [1,0,0])
        dp = v * uv
        
        # merge ...
        dp_n = np.random.normal(loc=dp, scale=dx_p)
        ddq_n  = random_quaternion(hmin=-dx_q, hmax=dx_q)
        dq_n = qmul(ddq_n, dq) # dq here is a global rotation.

        # compute motion updates + observation
        p1 = p + dp_n
        q1 = qmul(dq_n, q)

        dpr, dqr = qmath_np.xrel(p, q, p1, q1)
        obs.append([i-1, i, cat(dpr, dqr)])

        # apply motion updates
        p = p1
        q = q1

        # measure landmarks @ x1
        dzs = []
        rzs = []
        for zi, (zp,zq) in enumerate(zs):
            zp_n = np.random.normal(loc=zp, scale=dz_p)
            zdq  = random_quaternion(hmin=-dz_q, hmax=dz_q)
            zq_n = qmul(zdq, zq)

            zpr, zqr = qmath_np.xrel(p, q, zp_n, zq_n)

            #print '=== obs ==='
            #print zp, zq
            #print zp_n, zq_n
            #print qmath_np.xabs(p, q, zpr, zqr)
            #print '---     ---'

            # position index, landmark index, relative pose
            obs.append([i, n_t+zi, cat(zpr, zqr)])

    return obs, zs, (p,q)

def main():
    obs = gen_data(10, 4)
    print obs

if __name__ == "__main__":
    main()
