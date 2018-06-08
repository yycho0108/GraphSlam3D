import numpy as np
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

def qinv(q):
    """ Quaternion Inverse (Conjugate, Unit) """
    return tx.quaternion_conjugate(q)

def qmul(b,a):
    return tx.quaternion_multiply(b, a)

def qxv(q, v, norm=True):
    s = np.linalg.norm(v)
    if s < eps:
        return 0 * v
    v = v / s # make unit
    q_v = np.concatenate((v,[0]), axis=-1)
    q_c = qinv(q)
    v = qmul(qmul(q, q_v),q_c)[:-1] # == q.v.q^{-1}
    v = np.multiply(v,s)
    return v

def gen_data(n_t, n_l, motion_noise=1.0, measurement_noise=1.0):    
    n_s = 6 # state size : xyz, qlog(q)

    motion_noise = 1.0
    measurement_noise = 2.0

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

    hmax = np.deg2rad(10)
    dwmax = np.deg2rad(1.0)

    zqmax = np.deg2rad(60.0)

    dq = random_quaternion(hmin=-hmax, hmax=hmax)

    v = 1.0

    obs = []

    for i in range(1, n_t):
        # measure motion x0->x1
        ddq = random_quaternion(hmin=-dwmax, hmax=dwmax)
        dq = qmul(ddq, dq)

        # position; assume forward (x-axis) motion
        uv = qxv(q, [1,0,0])
        dp = v * uv
        
        # merge ...
        dp_n = np.random.normal(loc=dp, scale=motion_noise)
        ddq_n  = random_quaternion(hmin=-zqmax, hmax=zqmax)
        dq_n = qmul(ddq_n, dq) # dq here is a global rotation.

        # compute motion updates + observation
        q1 = qmul(dq, q)
        dp_n_rel= qxv(qinv(q), dp_n)
        dq_n_rel= qmul(qinv(q), qmul(dq_n, q))
        obs.append([i-1, i, cat(dp_n_rel, dq_n_rel)])

        # apply motion updates
        p += dp
        q  = q1

        # measure landmarks @ x1
        dzs = []
        rzs = []
        for zi, (zp,zq) in enumerate(zs):
            zp_n = np.random.normal(loc=zp, scale=measurement_noise)
            zdq  = random_quaternion(hmin=-zqmax, hmax=zqmax)
            zq_n = qmul(zdq, zq)

            zp_rel = qxv(qinv(q), zp_n - p)
            zq_rel = qmul(qinv(q), zq)

            # position index, landmark index, relative pose
            obs.append([i, n_t+zi, cat(zp_rel, zq_rel)])

    return obs

def main():
    obs = gen_data(10, 4)
    print obs

if __name__ == "__main__":
    main()
