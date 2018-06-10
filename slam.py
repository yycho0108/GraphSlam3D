import numpy as np
import sympy as sm
from tf import transformations as tx

from gen_data import gen_data
#from qmath import *
from utils import qmath_np
import pickle

eps = np.finfo(float).eps

def symbols(*args):
    return sm.symbols(*args, real=True)

class Report():
    def __init__(self, name):
        self._name = name
        self._title = ' {} '.format(self._name)

    def __enter__(self):
        print ''
        print '=' * 8 + self._title  + '=' * 8

    def __exit__(self, *args):
        print '-' * 8 + ' '*len(self._title) + '-' * 8
        print ''

class GraphSlam3(object):
    def __init__(self):
        #syms, Aij, Bij, eij = self._build()
        #self._syms = syms
        #self._Aij = Aij
        #self._Bij = Bij
        #self._eij = eij
        self._nodes = {}

    @staticmethod
    def Hadd(H, dH, i, j):
        if H[i][j] is None:
            H[i][j] = dH
        else:
            H[i][j] += dH

    def add_node(self, x, i):
        self._nodes[i] = x

    def add_edge(self, x, i0, i1):
        n = self._nodes

        # numpy version
        p0, q0 = qmath_np.x2pq(n[i0])
        p1, q1 = qmath_np.x2pq(n[i1])
        dp, dq = qmath_np.x2pq(x)

        Aij = qmath_np.Aij(p0, p1, dp, q0, q1, dq)
        Bij = qmath_np.Bij(p0, p1, dp, q0, q1, dq)
        eij = qmath_np.eij(p0, p1, dp, q0, q1, dq)

        # sympy version
        # values = (n[i0], n[i1], x)
        # sargs = {}
        # for (ks, vs) in zip(self._syms, values):
        #     sargs.update({k:v for k,v in zip(ks,vs)})
        # Aij = self._Aij.subs(sargs)
        # Bij = self._Bij.subs(sargs)
        # eij = self._eij.subs(sargs)

        # convert to np to handle either cases
        Aij = np.array(Aij).astype(np.float64)
        Bij = np.array(Bij).astype(np.float64)
        eij = np.array(eij).astype(np.float64)

        return Aij, Bij, eij

    def run(self, zs, max_nodes):
        H0 = np.zeros((6,6), dtype=np.float64)
        b0 = np.zeros((6,1), dtype=np.float64)

        # NOTE : change M() accordingly
        c = 1.0

        for it in range(100):
            H = [[H0.copy() for _ in range(max_nodes)] for _ in range(max_nodes)]
            b = [b0.copy() for _ in range(max_nodes)]

            for (z0, z1, z) in zs:

                if z0 not in self._nodes:
                    # very first position, encoded with (z0 == z1)
                    assert(z0 == z1)
                    self._nodes[z0] = z
                    continue

                if z1 not in self._nodes:
                    # add initial guess
                    self._nodes[z1] = qmath_np.xadd_rel(self._nodes[z0], z, T=False)

                Aij, Bij, eij = self.add_edge(z, z0, z1)

                # TODO : incorporate measurement uncertainties
                H[z0][z0] += c * np.matmul(Aij.T, Aij)
                H[z0][z1] += c * np.matmul(Aij.T, Bij)
                H[z1][z0] += c * np.matmul(Bij.T, Aij)
                H[z1][z1] += c * np.matmul(Bij.T, Bij)
                b[z0]     += c * np.matmul(Aij.T, eij)
                b[z1]     += c * np.matmul(Bij.T, eij)

            H[0][0] += np.eye(6)

            H = np.block(H)
            b = np.concatenate(b, axis=0)
            
            dx = np.matmul(np.linalg.pinv(H), -b)
            dx = np.reshape(dx, [-1,6])
            delta = np.mean(np.square(dx))
            print('delta', delta) # --> to check for convergence
            #print 'dx0', dx[0]
            #dx[0] *= 0.0

            x = [self._nodes[k] for k in sorted(self._nodes.keys())]

            if it == 0:
                xp = x

            n_t = 200

            with Report('x-raw'):
                print 'initial pose'
                print x[0]
                print 'final pose'
                print x[n_t-1]
                print 'last landmark'
                print x[-1]

            with Report('x-est'):
                print 'initial pose'
                print qmath_np.xadd(x[0], dx[0])
                print 'final pose'
                print qmath_np.xadd(x[n_t-1], dx[n_t-1])
                print 'last landmark'
                print qmath_np.xadd(x[-1], dx[-1])

            # update
            for i in range(max_nodes):
                self._nodes[i] = qmath_np.xadd(self._nodes[i], dx[i])

            if delta < 1e-4:
                break

        x = [self._nodes[k] for k in sorted(self._nodes.keys())]
        return x, xp

        # TODO : implement online version, maybe adapt Sebastian Thrun's code related to online slam
        # where relationships regarding x_{i-1} can be folded into x_{i}

def main():
    #dx_p = 0.1 #
    #dx_q = np.deg2rad(10.0)
    #dz_p = 0.1
    #dz_p = np.deg2rad(10.0)

    s = 0.05 # 1.0 = 1m = 57.2 deg.
    dx_p = 2 * s
    dx_q = 2 * s
    dz_p = s
    dz_q = s

    n_t = 200 # timesteps
    n_l = 4 # landmarks

    np.set_printoptions(precision=4)
    with np.errstate(invalid='raise'):
        max_nodes = n_t + n_l
        slam = GraphSlam3()
        zs, zs_gt, xs_gt = gen_data(n_t, n_l, dx_p, dx_q, dz_p, dz_q)
        es, esr = slam.run(zs, max_nodes=max_nodes)

        xs_e = es[:n_t]
        zs_e = es[-n_l:]

        xsr_e = esr[:n_t]

        print 'final pose'
        print xs_gt[-1]

        print 'zs_gt'
        for z in zs_gt:
            print z
        print '=='


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
