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

            # position index, landmark index, relative pose
            obs.append([i, n_t+zi, cat(zpr, zqr)])

    return obs, zs, (p,q)

def main():
    obs = gen_data(10, 4)
    print obs

if __name__ == "__main__":
    main()
