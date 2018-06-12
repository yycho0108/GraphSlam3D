import numpy as np
from gen_data import DataGenerator

from utils import qmath_np
from slam import GraphSlam3
import pickle
from matplotlib import pyplot as plt

def cat(a,b):
    return np.concatenate([a,b], axis=-1)

class Report():
    """ Provides separation between print calls"""
    def __init__(self, name):
        self._name = name
        self._title = ' {} '.format(self._name)

    def __enter__(self):
        print ''
        print '=' * 8 + self._title  + '=' * 8

    def __exit__(self, *args):
        print '-' * 8 + ' '*len(self._title) + '-' * 8
        print ''

class Print1(object):
    """ Print something once at iteration n. """
    def __init__(self, n=0):
        self._i = 0
        self._n = n
    def __call__(self, *args):
        if (self._i == self._n):
            print(args)
        self._i += 1

def main():
    s = 0.05 # 1.0 = 1m = 57.2 deg.
    dx_p = 2 * s
    dx_q = 2 * s
    dz_p = s
    dz_q = s
    p_obs = 0.25 # probability of observation

    n_t = 100 # timesteps
    n_l = 4 # landmarks

    seed = np.random.randint(1e5)
    gen  = DataGenerator(n_t=n_t, n_l=n_l, scale=100.0)

    np.set_printoptions(precision=4)
    with np.errstate(invalid='raise'):
        max_nodes = n_t + n_l
        slam = GraphSlam3(n_l)

        # V2 : Online Version
        np.random.seed(seed)
        xs, zs, obs = gen(
                dx_p,dx_q,dz_p,dz_q,
                p_obs=p_obs,
                stepwise=True,
                seed=seed)

        with Report('Ground Truth'):
            print 'final pose'
            print xs[-1]
            print 'landmarks'
            print np.asarray(zs)

        # compute raw results ...
        xes_raw = []
        x_raw = cat(*xs[0])
        for dx, z in obs:
            x_raw = qmath_np.xadd_rel(x_raw, dx, T=False)
            xes_raw.append(x_raw.copy())

        with Report('Raw Results'):
            print 'final pose'
            print x_raw
            print 'landmarks'
            print '[Not available at this time]'

        xesr_p, xesr_q = zip(*[x for x in xs[1:]])
        xeso_p, xeso_q = zip(*[qmath_np.x2pq(x) for x in xes_raw])
        dp = np.subtract(xesr_p, xeso_p)
        dq = [qmath_np.T(qmath_np.qmul(q1, qmath_np.qinv(q0))) for (q1,q0) in zip(xesr_q, xeso_q)]
        delta  = np.concatenate([dp,dq], axis=-1)
        delta  = np.linalg.norm(delta, axis=-1)
        plt.plot(delta)

        # online slam ...
        slam.initialize(cat(*xs[0]))
        xes_onl = []
        for dx, z in obs:
            es = slam.step(dx, z)
            xes_onl.append(es[1].copy())

        with Report('Online'):
            print 'final pose'
            print es[1]
            print 'landmarks'
            print np.asarray(es[2:])

        xesr_p, xesr_q = zip(*[x for x in xs[1:]])
        xeso_p, xeso_q = zip(*[qmath_np.x2pq(x) for x in xes_onl])
        dp = np.subtract(xesr_p, xeso_p)
        dq = [qmath_np.T(qmath_np.qmul(q1, qmath_np.qinv(q0))) for (q1,q0) in zip(xesr_q, xeso_q)]
        delta  = np.concatenate([dp,dq], axis=-1)
        delta  = np.linalg.norm(delta, axis=-1)
        plt.plot(delta)

        # offline slam ...
        if n_t < 1000:
            # try not to do this when matrix will get too big
            np.random.seed(seed)
            xs, zs, obs = gen(
                    dx_p,dx_q,dz_p,dz_q,
                    p_obs=p_obs,
                    stepwise=False,
                    seed=seed)

            slam._nodes = {}
            slam.initialize(cat(*xs[0]))
            xes_off = slam.run(obs, max_nodes=max_nodes, n_iter=1)

            with Report('Offline'):
                print 'final pose'
                print xes_off[n_t-1]
                print 'landmarks'
                print np.asarray(es[-n_l:])

            xesr_p, xesr_q = zip(*[x for x in xs[1:]])
            xeso_p, xeso_q = zip(*[qmath_np.x2pq(x) for x in xes_off[1:n_t]])
            dp = np.subtract(xesr_p, xeso_p)
            dq = [qmath_np.T(qmath_np.qmul(q1, qmath_np.qinv(q0))) for (q1,q0) in zip(xesr_q, xeso_q)]
            delta  = np.concatenate([dp,dq], axis=-1)
            delta  = np.linalg.norm(delta, axis=-1)

            plt.plot(delta)

        plt.legend(['raw','slam-on', 'slam-off'])
        plt.title('Estimated Error Over Time')
        plt.show()

        return

        ps, qs = zip(*xs_gt)
        zs     = [zs_gt for _ in range(n_t)] # repeat
        ep, eq = zip(*[qmath_np.x2pq(e) for e in xs_e])
        rp, rq = zip(*[qmath_np.x2pq(e) for e in xsr_e])
        ezs    = [qmath_np.x2pq(e) for e in zs_e]
        ezs    = [ezs for _ in range(n_t)]
        #rzs    = zs

        #print len(ps)
        #print len(qs)
        #print len(zs_gt)
        #print len(ep)
        #print len(eq)
        #print len(ezs)
        #print len(zs)


        with open('/tmp/data.pkl', 'w+') as f:
            pickle.dump(zip(*[ps,qs,zs,ep,eq,rp,rq,ezs]), f)
        
        #(p,q,zs,ep,eq,ezs,rzs)


if __name__ == "__main__":
    main()
