"""
Attempt to implement Graph Slam in 3-Dimensions.
"""
import numpy as np
import sympy as sm
import scipy.sparse
import scipy.sparse.linalg

from tf import transformations as tx

from gen_data import gen_data
from gen_data import gen_data_stepwise

#from qmath import *
from utils import qmath_np
import pickle

eps = np.finfo(float).eps

def cat(a,b):
    return np.concatenate([a,b], axis=-1)

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

def block(ar):
    """ Convert Block Matrix to Dense Matrix """
    ni,nj,nk,nl = ar.shape
    return np.swapaxes(ar, 1, 2).reshape(ni*nk, nj*nl)

def unblock(ar, nknl):
    """ Convert Dense Matrix to Block Matrix """
    nk,nl = nknl
    nink, njnl = ar.shape
    ni = nink/nk
    nj = njnl/nl
    return ar.reshape(ni,nk,nj,nl).swapaxes(1,2)

class GraphSlam3(object):
    def __init__(self, n_l):
        self._nodes = {}
        self._n_l = n_l

    def add_edge(self, x, i0, i1):
        n = self._nodes

        # numpy version
        p0, q0 = qmath_np.x2pq(n[i0])
        p1, q1 = qmath_np.x2pq(n[i1])
        dp, dq = qmath_np.x2pq(x)

        Aij = qmath_np.Aij(p0, p1, dp, q0, q1, dq)
        Bij = qmath_np.Bij(p0, p1, dp, q0, q1, dq)
        eij = qmath_np.eij(p0, p1, dp, q0, q1, dq)

        # convert to np to handle either cases
        Aij = np.array(Aij).astype(np.float64)
        Bij = np.array(Bij).astype(np.float64)
        eij = np.array(eij).astype(np.float64)

        return Aij, Bij, eij

    def initialize(self, x0):
        n = 2 + self._n_l
        self._H = np.zeros((n,n,6,6), dtype=np.float64)
        self._b = np.zeros((n,1,6,1), dtype=np.float64)
        self._H[0,0] = np.eye(6)
        #self._nodes[0] = np.asarray([0,0,0,0,0,0,1])
        self._nodes[0] = x0

        # TODO : check if this is valid vv
        #p, q = qmath_np.x2pq(x0)
        #x = np.concatenate([p,qmath_np.T(q)], axis=-1)
        #self._b[0,0,:,0] = x

    def step(self, x=None, zs=None):
        """ Online Version, does not work """
        c = 1.0

        # " expand "
        self._H[1,:] = 0.0
        self._H[:,1] = 0.0
        self._b[1]   = 0.0

        # apply motion updates first
        # TODO : check if this is even valid
        self._nodes[1] = qmath_np.xadd_rel(self._nodes[0], x, T=False)
        Aij, Bij, eij = self.add_edge(x, 0, 1)

        self._H[0,0] += c * np.matmul(Aij.T, Aij)
        self._H[0,1] += c * np.matmul(Aij.T, Bij)
        self._H[1,0] += c * np.matmul(Bij.T, Aij)
        self._H[1,1] += c * np.matmul(Bij.T, Bij)
        self._b[0]   += c * np.matmul(Aij.T, eij)
        self._b[1]   += c * np.matmul(Bij.T, eij)

        # H and b are organized as (X0, X1, L0, X1, ...)
        # Such that H[0,..] pertains to X0, and so on.

        # now with observations ...
        for (z0, z1, z) in zs:
            if z1 not in self._nodes:
                # initial guess
                self._nodes[z1] = qmath_np.xadd_rel(
                        self._nodes[z0], z, T=False)
                #print self._nodes[z1]
                continue
            Aij, Bij, eij = self.add_edge(z, z0, z1) # considered observed @ 1
            self._H[z0,z0] += c * np.matmul(Aij.T, Aij)
            self._H[z0,z1] += c * np.matmul(Aij.T, Bij)
            self._H[z1,z0] += c * np.matmul(Bij.T, Aij)
            self._H[z1,z1] += c * np.matmul(Bij.T, Bij)
            self._b[z0]   += c * np.matmul(Aij.T, eij)
            self._b[z1]   += c * np.matmul(Bij.T, eij)

        H00 = block(self._H[:1,:1])
        H01 = block(self._H[:1,1:])
        H11 = block(self._H[1:,1:])

        B00 = block(self._b[:1,:1])
        B10 = block(self._b[1:,:1])

        AtBi = np.matmul(H01.T, np.linalg.pinv(H00))
        XiP  = B10
        #print np.mean(np.abs(self._H))

        # fold previous information into new matrix
        H = H11 + np.matmul(AtBi, H01) # TODO : ??????????????????? Why does this work?
        B = B10 - np.matmul(AtBi, B00)

        dx = np.matmul(np.linalg.pinv(H), -B)
        dx = np.reshape(dx, [-1,6]) # [x1, l0, ... ln]
        for i in range(1, 2+self._n_l):
            self._nodes[i] = qmath_np.xadd(self._nodes[i], dx[i-1])

        #dx2 = np.matmul(np.linalg.pinv(block(self._H)), -block(self._b))
        #dx2 = np.reshape(dx2, [-1,6])
        ##print 'dx2', dx2[1:]
        #for i in range(0, 2+self._n_l):
        #    self._nodes[i] = qmath_np.xadd(self._nodes[i], dx2[i])

        # replace previous node with current position
        self._nodes[0] = self._nodes[1].copy()

        H = unblock(H, (6,6))
        B = unblock(B, (6,1))

        # assign at appropriate places, with x_0 being updated with x_1
        self._H[:1,:1] = H[:1,:1]
        self._H[:1,2:] = H[:1,1:]
        self._H[2:,:1] = H[1:,:1]
        self._H[2:,2:] = H[1:,1:]
        self._b[:1] = B[:1]
        self._b[2:] = B[1:]

        x = [self._nodes[k] for k in sorted(self._nodes.keys())]
        return x

    def run(self, zs, max_nodes):
        """ Offline version, works. """
        H0 = np.zeros((6,6), dtype=np.float64)
        b0 = np.zeros((6,1), dtype=np.float64)

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
            # opt 1
            dx = np.matmul(np.linalg.pinv(H), -b)

            # opt 2
            #Hi = scipy.sparse.csr_matrix(H)
            #dx = np.asarray(scipy.sparse.linalg.spsolve(Hi, b))

            dx = np.reshape(dx, [-1,6])
            delta = np.mean(np.square(dx))
            #print 'dx0', dx[0]
            #dx[0] *= 0.0

            x = [self._nodes[k] for k in sorted(self._nodes.keys())]

            if it == 0:
                xp = x

            n_t = 2

            #with Report('x-raw'):
            #    print 'initial pose'
            #    print x[0]
            #    print 'final pose'
            #    print x[n_t-1]
            #    print 'last landmark'
            #    print x[-1]

            #with Report('x-est'):
            #    print 'initial pose'
            #    print qmath_np.xadd(x[0], dx[0])
            #    print 'final pose'
            #    print qmath_np.xadd(x[n_t-1], dx[n_t-1])
            #    print 'last landmark'
            #    print qmath_np.xadd(x[-1], dx[-1])

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

    n_t = 100 # timesteps
    n_l = 20 # landmarks

    np.set_printoptions(precision=4)
    with np.errstate(invalid='raise'):
        max_nodes = n_t + n_l
        slam = GraphSlam3(n_l)

        # V2 : Online Version
        np.random.seed(0)
        xs, zs, obs = gen_data_stepwise(n_t, n_l, dx_p, dx_q, dz_p, dz_q)

        slam.initialize(cat(*xs[0]))
        x_raw = cat(*xs[0])
        for dx, z in obs:
            es = slam.step(dx, z)
            x_raw = qmath_np.xadd_rel(x_raw, dx, T=False)

        print 'final pose (raw)'
        print x_raw

        print 'final pose (online)'
        print es[1]

        # V1 : Offline Version
        np.random.seed(0)
        slam._nodes = {}
        zsr, zs, xs = gen_data(n_t, n_l, dx_p, dx_q, dz_p, dz_q)
        es, esr = slam.run(zsr, max_nodes=max_nodes)

        print 'final pose (offline)'
        print es[n_t-1]

        xs_e = es[:n_t]
        zs_e = es[-n_l:]

        #xsr_e = esr[:n_t]

        print 'final pose (ground truth)'
        print xs[-1]

        print 'zs Ground Truth'
        for z in zs:
            print z
        print '=='

        #print np.asarray(xs_e)
        #print np.asarray(zs_e)
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
