import numpy as np
from utils import qmath_np

qinv = qmath_np.qinv
qmul = qmath_np.qmul
qxv  = qmath_np.qxv

from tf import transformations as tx

eps = np.finfo(float).eps

def cat(a,b):
    return np.concatenate([a,b], axis=-1)

class DataGenerator(object):
    """ Note : Is not actually a python generator. """

    def __init__(self, n_t, n_l,
            scale=50.0,
            v = 2.0,
            dw = np.deg2rad(3.0)
            ):
        self._n_t = n_t
        self._n_l = n_l
        self._scale = scale

        self._v = v
        self._dwmax = dw

    def reset(self):
        # initial pose
        p = qmath_np.rt(s=self._scale)
        q = qmath_np.rq(s=np.pi)

        # TODO : configurable motion parameters
        w = qmath_np.rq(s=np.deg2rad(10))

        # initial landmarks
        zs = []
        for zi in range(self._n_l):
            zp = qmath_np.rt(s=self._scale)
            zq = qmath_np.rq(s=np.pi)
            zs.append([zp,zq])

        return p, q, w, zs

    def __call__(self,
            dx_p  = 1.0, # noise params
            dx_q  = 1.0,
            dz_p  = 1.0,
            dz_q  = 1.0,
            p_obs = 1.0,
            stepwise=True,
            seed = None
            ):
        """
        Returns The Following:

        - Ground Truth Trajectory
        - Ground Truth Landmark Location
        - Pose Update Observations
        - Raw Landmark Observations

        """
        if seed is not None:
            np.random.seed(seed)

        p,q,w,zs = self.reset()
        v = self._v

        xs = [] # ground truth trajectory
        dxs = [] # Pose Update Observations
        obs = [] # Raw Landmark Observations
        # TODO : configurable motion parameters

        for i in range(1, self._n_t):
            xs.append( (p.copy(), q.copy()) )

            # compute motion ...
            dw  = qmath_np.rq(s=self._dwmax)
            w   = qmul(dw, w)
            uv  = qxv(q, [1,0,0])
            dp  = v * uv

            # add noise ...
            dp_n = np.random.normal(loc=dp, scale=dx_p)
            dw_n = qmath_np.rq(s=dx_q)
            w_n  = qmul(dw_n, w)

            # noisy motion observation
            p1 = p + dp_n
            q1 = qmul(w_n, q)

            dpr, dqr = qmath_np.xrel(p, q, p1, q1)

            if not stepwise:
                obs.append([i-1, i, cat(dpr,dqr)])

            # ~= xadd_abs
            p = p + dp
            q = qmul(w, q)

            obs_z = []
            for zi, (zp,zq) in enumerate(zs):
                if np.random.random() > p_obs:
                    continue
                zp_n = np.random.normal(loc=zp, scale=dz_p)
                zdq  = qmath_np.rq(s=dz_q)
                zq_n = qmul(zdq, zq)
                zpr, zqr = qmath_np.xrel(p, q, zp_n, zq_n)

                if stepwise:
                    obs_z.append([1, 2+zi, cat(zpr,zqr)])
                else:
                    obs.append([i, self._n_t+zi, cat(zpr,zqr)])

            if stepwise:
                obs_i = [cat(dpr,dqr), obs_z]
                obs.append(obs_i)

        xs.append((p.copy(), q.copy()))

        return xs, zs, obs

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

    # === motion parameters ===
    hmax = np.deg2rad(10)
    dwmax = np.deg2rad(1.0)
    v = 5.0
    # =========================

    dq = np.random.uniform(low=-hmax, high=hmax)

    obs = []

    # SPECIAL : add initial pose
    obs.append([0, 0, cat(p0, q0)])

    xs = []

    for i in range(1, n_t):
        xs.append((p.copy(), q.copy()))

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

    xs.append((p.copy(), q.copy())) # ground-truth motion data

    # TODO : below is confusing, fix later to fit api, etc.

    return obs, zs, xs

def main():
    obs = gen_data_2d(10, 4)
    gen = DataGenerator(10,4)
    xs, zs, obs = gen()
    print obs

if __name__ == "__main__":
    main()
