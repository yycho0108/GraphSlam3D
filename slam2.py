import numpy as np
import sympy as sm
import scipy.linalg
from tf import transformations as tx

from gen_data import gen_data_2d
#from qmath import *
from utils import qmath_np

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

class GraphSlam2(object):
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
        p0, q0 = qmath_np.x2pq2(n[i0])
        p1, q1 = qmath_np.x2pq2(n[i1])
        dp, dq = qmath_np.x2pq2(x)

        Aij = qmath_np.Aij2(p0, p1, dp, q0, q1, dq)
        Bij = qmath_np.Bij2(p0, p1, dp, q0, q1, dq)
        eij = qmath_np.eij2(p0, p1, dp, q0, q1, dq)

        return Aij, Bij, eij

    def run(self, zs, max_nodes):
        H0 = np.zeros((3,3), dtype=np.float32)
        b0 = np.zeros((3,1), dtype=np.float32)
        omega = 1.0 * np.eye(3)

        # NOTE : change M() accordingly

        # start with initial guess
        #for xi, x in enumerate(xs):
        #    self._nodes[i] = x

        for _ in range(1):
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
                    self._nodes[z1] = qmath_np.xadd_rel2(self._nodes[z0], z, T=False)

                Aij, Bij, eij = self.add_edge(z, z0, z1)

                # TODO : incorporate measurement uncertainties
                H[z0][z0] += Aij.T.dot(omega).dot(Aij)
                H[z0][z1] += Aij.T.dot(omega).dot(Bij)#np.matmul(Aij.T, Bij)
                H[z1][z0] += Bij.T.dot(omega).dot(Aij)#np.matmul(Bij.T, Aij)
                H[z1][z1] += Bij.T.dot(omega).dot(Bij)#np.matmul(Bij.T, Bij)
                b[z0]     += Aij.T.dot(omega).dot(eij)#np.matmul(Aij.T, eij)
                b[z1]     += Bij.T.dot(omega).dot(eij)#np.matmul(Bij.T, eij)

            H[0][0] += np.eye(3)
            #print np.asarray([[np.abs(np.sum(e))>0 for e in r] for r in H], dtype=np.int32)

            H = np.block(H)
            #print (np.abs(H) > 0).astype(np.int32)
            b = np.concatenate(b, axis=0)
            
            dx = np.matmul(np.linalg.pinv(H), -b)
            #dx = scipy.linalg.solve(H, -b)

            #with Report('dx'):
            #    print dx[:5]
            #    print dx2[:5]

            dx = np.reshape(dx, [-1,3])
            #dx[0] *= 0.0

            x = [self._nodes[k] for k in sorted(self._nodes.keys())]
            n_t = 100

            with Report('x-raw'):
                print 'initial pose'
                print x[0]
                print 'final pose'
                print x[n_t-1]
                print 'last landmark'
                print x[-1]

            with Report('x-est'):
                print 'initial pose'
                print qmath_np.xadd2(x[0], dx[0])
                print 'final pose'
                print qmath_np.xadd2(x[n_t-1], dx[n_t-1])
                print 'last landmark'
                print qmath_np.xadd2(x[-1], dx[-1])

            # update
            for i in range(max_nodes):
                self._nodes[i] = qmath_np.xadd2(self._nodes[i], dx[i])

        # TODO : implement online version, maybe adapt Sebastian Thrun's code related to online slam
        # where relationships regarding x_{i-1} can be folded into x_{i}

def main():
    #dx_p = 0.1 #
    #dx_q = np.deg2rad(10.0)
    #dz_p = 0.1
    #dz_p = np.deg2rad(10.0)

    s = 0.1 # 1.0 = 1m = 57.2 deg.
    dx_p = 10*s
    dx_q = s
    dz_p = 10*s
    dz_q = s

    n_t = 100 # timesteps
    n_l = 4 # landmarks

    np.set_printoptions(precision=4)
    with np.errstate(invalid='raise'):
        max_nodes = n_t + n_l
        slam = GraphSlam2()
        zs, zs_gt, (p,q) = gen_data_2d(n_t, n_l, dx_p, dx_q, dz_p, dz_q)
        slam.run(zs, max_nodes=max_nodes)

        print 'final pose'
        print p, q

        print 'zs_gt'
        for z in zs_gt:
            print z
        print '=='

if __name__ == "__main__":
    main()
