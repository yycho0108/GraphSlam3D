import numpy as np
from utils import qmath_np

qinv = qmath_np.qinv
qmul = qmath_np.qmul
qxv  = qmath_np.qxv

from tf import transformations as tx

eps = np.finfo(float).eps

def cat(a,b):
    return np.concatenate([a,b], axis=-1)

def gen_data(n_t, n_l,
        dx_p = 1.0,
        dx_q = 1.0,
        dz_p = 1.0,
        dz_q = 1.0
        ):    
    graph = []

    # initial pose, (x0, q0)
    p0, q0 = qmath_np.rt(s=50.0), qmath_np.rq(s=np.pi)
    x0  = cat(p0, q0)
    graph.append([0, 0, x0])

    zs = []
    for zi in range(n_l):
        zp, zq = qmath_np.rt(s=50.0), qmath_np.rq(s=np.pi)
        zs.append([zp, zq])

    p, q = p0.copy(), q0.copy()
    print 'initial'
    print p, q

    # === motion parameters ===
    hmax = np.deg2rad(10)
    dwmax = np.deg2rad(1.0)
    v = 1.0
    # =========================

    dq = qmath_np.rq(s=hmax)

    obs = []

    # SPECIAL : add initial pose
    obs.append([0, 0, cat(p0, q0)])

    for i in range(1, n_t):
        # measure motion x0->x1
        ddq = qmath_np.rq(s=dwmax)
        dq = qmul(ddq, dq)

        # position; assume forward (x-axis) motion
        uv = qxv(q, [1,0,0])
        dp = v * uv
        
        # merge ...
        dp_n = np.random.normal(loc=dp, scale=dx_p)
        ddq_n  = qmath_np.rq(s=dx_q)
        dq_n = qmul(ddq_n, dq) # dq here is a global rotation.

        # compute motion updates + observation
        p1 = p + dp_n
        q1 = qmul(dq_n, q)

        dpr, dqr = qmath_np.xrel(p, q, p1, q1)
        obs.append([i-1, i, cat(dpr, dqr)])

        # apply motion updates
        p = p + dp
        q = qmul(dq, q)

        # measure landmarks @ x1
        for zi, (zp,zq) in enumerate(zs):
            zp_n = np.random.normal(loc=zp, scale=dz_p)
            zdq  = qmath_np.rq(s=dz_q)
            zq_n = qmul(zdq, zq)

            zpr, zqr = qmath_np.xrel(p, q, zp_n, zq_n)

            #print 'obs'
            #print zp, zq
            #print qmath_np.x2pq(qmath_np.xadd_rel(cat(p, q), cat(zpr, zqr), T=False))
            #print '=='

            # position index, landmark index, relative pose
            obs.append([i, n_t+zi, cat(zpr, zqr)])

    return obs, zs, (p,q)

def gen_data_2d(n_t, n_l,
        dx_p = 1.0,
        dx_q = 1.0,
        dz_p = 1.0,
        dz_q = 1.0
        ):    
    graph = []

    # initial pose, (x0, q0)
    p0 = np.random.uniform(-50.0, 50.0, size=2)
    p0 *= 0.0
    q0 = np.random.uniform(-np.pi, np.pi, size=1)
    x0  = cat(p0, q0)
    graph.append([0, 0, x0])

    zs = []
    for zi in range(n_l):
        zp = np.random.uniform(-50.0, 50.0, size=2)
        zq = np.random.uniform(-np.pi, np.pi)
        zs.append([zp, zq])

    p, q = p0.copy(), q0.copy()
    print 'initial'
    print p, q

    # === motion parameters ===
    hmax = np.deg2rad(10)
    dwmax = np.deg2rad(1.0)
    v = 1.0
    # =========================

    dq = np.random.uniform(low=-hmax, high=hmax)

    obs = []

    # SPECIAL : add initial pose
    obs.append([0, 0, cat(p0, q0)])

    for i in range(1, n_t):
        # measure motion x0->x1
        ddq = qmath_np.rq2(dwmax)
        dq  += ddq

        # position; assume forward (x-axis) motion
        uv = qmath_np.q2R2(q).dot([1,0])
        dp = v * uv
        
        # merge ...
        dp_n = np.random.normal(loc=dp, scale=dx_p)
        ddq_n = qmath_np.rq2(dx_q)
        dq_n = dq + ddq_n

        # compute motion updates + observation
        p1 = p + dp_n
        q1 = q + dq_n

        dpr, dqr = qmath_np.xrel2(p, q, p1, q1)
        obs.append([i-1, i, cat(dpr, dqr)])

        # apply motion updates
        p = p + dp
        q = qmath_np.hnorm2(q + dq)

        # measure landmarks @ x1
        for zi, (zp,zq) in enumerate(zs):
            zp_n = np.random.normal(loc=zp, scale=dz_p)
            zdq  = qmath_np.rq2(s=dz_q)
            zq_n = zq + zdq

            zpr, zqr = qmath_np.xrel2(p, q, zp_n, zq_n)

            #print 'obs'
            #print zp, zq
            #print qmath_np.x2pq2(qmath_np.xadd_rel2(cat(p, q), cat(zpr, zqr), T=False))
            #print '=='

            # position index, landmark index, relative pose
            obs.append([i, n_t+zi, cat(zpr, zqr)])

    return obs, zs, (p,q)

def main():
    #obs = gen_data(10, 4)
    obs = gen_data_2d(10, 4)
    print obs

if __name__ == "__main__":
    main()
