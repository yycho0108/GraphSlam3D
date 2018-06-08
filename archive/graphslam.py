import numpy as np
from matplotlib import pyplot as plt
import tf.transformations as tx
import pickle

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

#def dq(qa, qb):
#    # expression for difference between two quaternions
#    d = qmul(qa, qinv(qb))
#    return qlog(d)

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

class GraphSlam(object):
    def __init__(self, n_s, n_l, q=1.0, r=1.0):
        n_r = n_s * (n_l + 2) # 2 = xprv, xnxt
        self._n_s = n_s
        self._n_l = n_l
        self._omega = np.zeros(shape=(n_r, n_r), dtype=np.float32)
        self._xi = np.zeros(shape=(n_r,1), dtype=np.float32)
        self._w_q = 1.0 / q # motion noise
        self._w_r = 1.0 / r # meausrement noise

    @staticmethod
    def add_relationship(o, x, i0, i1, v, w=1.0):
        # good to set w = (1.0 / noise)
        o[i0,i0] += w
        o[i1,i1] += w
        o[i0,i1] += -w
        o[i1,i0] += -w
        x[i0,0] += -v*w
        x[i1,0] += v*w

    def initialize(self, x, w=1.0):
        # x = (x,y, ...)
        # initialize with position
        # organized as (L, X1, X0)
        i0 = 0#(self._n_l+1) * self._n_s

        # assert len(x) == self._n_s
        idx = range(i0, i0 + self._n_s)
        self._omega[idx, idx] = w
        self._xi[idx, 0] = x

    def step(self, x=None, z=None, step=0):
        n_l = self._n_l
        n_s = self._n_s

        # " expand "
        self._omega[n_s:2*n_s, :] = 0.0
        self._omega[:, n_s:2*n_s] = 0.0
        self._xi[n_s:2*n_s] = 0.0

        for i in range(n_s):
            # add landmark information
            for z_i, obs in enumerate(z):
                self.add_relationship(
                        self._omega,
                        self._xi,
                        i, # x_0
                        (z_i+2)*n_s + i,  # l_i
                        obs[i],
                        self._w_r)

            # add motion information
            self.add_relationship(
                    self._omega,
                    self._xi,
                    i, # x_0
                    n_s + i, # x_1
                    x[i],
                    self._w_q)
        
        # omega organized as
        # omega = (X0, X1, L)

        op = self._omega[n_s:, n_s:] # L, X1
        a  = self._omega[:n_s, n_s:] # x0, (l,X1)
        b  = self._omega[:n_s, :n_s] # x0, x0
        c  = self._xi[:n_s, :1] # x0

        atbi = np.matmul(a.T, np.linalg.pinv(b)) #(l+x1), x0
        xip  = self._xi[n_s:, :1] # (l,x1)

        # == below contains (L, X1)
        omega = op - np.matmul(atbi, a)
        xi    = xip - np.matmul(atbi, c)
        mu    = np.matmul(np.linalg.pinv(omega), xi)

        # re-form omega/xi with x1 as x0
        self._omega[:n_s, :n_s] = omega[:n_s, :n_s] # x0
        self._omega[:n_s, 2*n_s:] = omega[:n_s, n_s:] # x0
        self._omega[2*n_s:, :n_s] = omega[n_s:, :n_s]
        self._omega[2*n_s:, 2*n_s:] = omega[n_s:, n_s:] # landmarks
        self._xi[:n_s, 0] = xi[:n_s, 0]
        self._xi[2*n_s:, 0] = xi[n_s:, 0]

        return mu

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

def parametrize(x, q):
    qv = qlog(q)
    return np.concatenate([x,qv], axis=-1)

def unparametrize(p):
    x = p[:3]
    qv = p[3:]
    q = qvexp(qv)
    return x, q

def delta(x0, q0, x1, q1):
    #dx = x1 - x0
    #dq = qlog(q1) - qlog(q0)
    #return np.concatenate([dx,dq], axis=-1)
    return parametrize(x0,q0) - parametrize(x1,q1)

def main():
    n_s = 6 # state size : xyz, qlog(q)
    n_l = 5 # number of landmarks
    n_t = 500 # number of timesteps

    motion_noise = 1.0
    measurement_noise = 2.0

    # initial pose, (x0, q0)
    p0, q0 = random_pose()
    p0 *= 0.0
    x0 = parametrize(p0,q0)

    zs = []
    for zi in range(n_l):
        zp, zq = random_pose()
        zs.append([zp, zq])

    slam = GraphSlam(n_s=n_s, n_l=n_l, q=motion_noise,r=measurement_noise)
    slam.initialize(x=x0, w=1.0)

    p, q = p0.copy(), q0.copy()

    hmax = np.deg2rad(10)
    dwmax = np.deg2rad(1.0)

    zqmax = np.deg2rad(10.0)

    dq = random_quaternion(hmin=-hmax, hmax=hmax)

    v = 1.0

    gt_x = []
    est_x = []
    est_z = []
    data = []
    
    for i in range(n_t):
        # measure landmarks @ x0
        dzs = []
        rzs = []
        for (zp,zq) in zs:
            zp_n = np.random.normal(loc=zp, scale=measurement_noise)
            zdq  = random_quaternion(hmin=-zqmax, hmax=zqmax)
            zq_n = qmul(zdq, zq)
            dz = parametrize(zp_n,zq_n) - parametrize(p,q)
            dzs.append(dz)
            rzs.append([zp_n, zq_n])

        dz = np.float32(dzs)
        dz_n = np.copy(dz)

        #dz_n = np.random.normal(loc=dz, scale=measurement_noise) # TODO : is this ok? hmm

        # measure motion x0->x1
        ddq = random_quaternion(hmin=-dwmax, hmax=dwmax)
        dq = qmul(ddq, dq)

        # position; assume forward (x-axis) motion
        uv = qxv(q, [1,0,0])
        dp = v * uv
        
        # merge ...
        dx = parametrize(dp, dq)
        dp_n = np.random.normal(loc=dp, scale=motion_noise)
        ddq_n  = random_quaternion(hmin=-zqmax, hmax=zqmax)
        dq_n = qmul(ddq_n, dq)

        #dx_n = np.copy(dx)
        dx_n = parametrize(dp_n, dq_n)
        #dx_n = np.random.normal(loc=dx, scale=motion_noise) # TODO : is this ok? hmm
        dx_n = np.ravel(dx_n)

        e = slam.step(x=dx_n, z=dz_n, step=i)
        
        e = np.reshape(e, [-1, 6])
        ep, eq = unparametrize(e[0])
        ez = [unparametrize(_e) for _e in e[1:]]

        # apply motion updates
        p += dp
        q  = qmul(dq, q)

        gt_x.append([p,q])
        est_x.append([ep, eq])
        est_z.append(ez)

        d = [np.copy(e) for e in [p,q,zs,ep,eq,ez,rzs]]
        data.append(d)

    with open('/tmp/data.pkl', 'w+') as f:
        pickle.dump(data, f)

    print ' == landmarks == '
    for (zp,zq) in zs:
        print zp, zq
    print ' == position == '
    print p, q

    print ' == estimated landmarks == '
    for ezp, ezq in ez:
        print ezp, ezq

    print ' == estimated position == '
    print ep, eq

if __name__ == "__main__":
    main()
