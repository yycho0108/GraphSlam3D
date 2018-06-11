import numpy as np
from gen_data import DataGenerator

from utils import qmath_np
from slam import GraphSlam3
import pickle

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
    p_obs = 0.5 # probability of observation

    n_t = 200 # timesteps
    n_l = 4 # landmarks

    seed = np.random.randint(1e5)
    gen  = DataGenerator(n_t=n_t, n_l=n_l)

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
        x_raw = cat(*xs[0])
        for dx, z in obs:
            x_raw = qmath_np.xadd_rel(x_raw, dx, T=False)

        with Report('Raw Results'):
            print 'final pose'
            print x_raw
            print 'landmarks'
            print '[Not available at this time]'

        # online slam ...
        slam.initialize(cat(*xs[0]))
        for dx, z in obs:
            x_raw = qmath_np.xadd_rel(x_raw, dx, T=False)
            es = slam.step(dx, z)

        with Report('Online'):
            print 'final pose'
            print es[1]
            print 'landmarks'
            print np.asarray(es[2:])

        # offline slam ...
        np.random.seed(seed)
        slam._nodes = {}
        xs, zs, obs = gen(
                dx_p,dx_q,dz_p,dz_q,
                p_obs=p_obs,
                stepwise=False,
                seed=seed)

        slam.initialize(cat(*xs[0]))
        es = slam.run(obs, max_nodes=max_nodes)

        with Report('Offline'):
            print 'final pose'
            print es[n_t-1]
            print 'landmarks'
            print np.asarray(es[-n_l:])

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
