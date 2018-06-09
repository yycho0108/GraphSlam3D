import numpy as np
import sympy as sm
from tf import transformations as tx

from gen_data import gen_data
from qmath import *
from utils import qmath_np

eps = np.finfo(float).eps

def symbols(*args):
    return sm.symbols(*args, real=True)

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
        Aij = np.array(Aij).astype(np.float32)
        Bij = np.array(Bij).astype(np.float32)
        eij = np.array(eij).astype(np.float32)

        return Aij, Bij, eij

    def run(self, zs, max_nodes):
        H0 = np.zeros((6,6), dtype=np.float32)# TODO : check if this needs to be created underneath
        b0 = np.zeros((6,1), dtype=np.float32)

        H = [[H0.copy() for _ in range(max_nodes)] for _ in range(max_nodes)]
        b = [b0.copy() for _ in range(max_nodes)]

        for (z0, z1, z) in zs:
            if z0 not in self._nodes:
                assert(z0 == z1)
                self._nodes[z0] = z
                continue

            if z1 not in self._nodes:
                # add initial guesses
                #print z0, z1, self._nodes[z0], z
                self._nodes[z1] = qmath_np.xadd(self._nodes[z0], z, T=False)
                #print self._nodes[z1]

            Aij, Bij, eij = self.add_edge(z, z0, z1)

            try:
                H[z0][z0] += np.matmul(Aij.T, Aij)
                H[z0][z1] += np.matmul(Aij.T, Bij)
                H[z1][z0] += np.matmul(Bij.T, Aij)
                H[z1][z1] += np.matmul(Bij.T, Bij)
                b[z0]     += np.matmul(Aij.T, eij)
                b[z1]     += np.matmul(Bij.T, eij)
            except Exception as e:
                print 'Waat? [z0:{},z1:{}] : {}'.format(z0, z1, e)

        H[0][0] += np.eye(6) # TODO : necessary?

        H = np.block(H)
        b = np.concatenate(b, axis=0)
        
        #print 'H', H
        #print 'b', b

        dx = np.matmul(np.linalg.pinv(H), -b)
        dx = np.reshape(dx, [-1,6])
        #print 'dx', dx

        #np.save('dx.npy', dx)

        x = [self._nodes[k] for k in sorted(self._nodes.keys())]

        print 'x-raw'
        print x[0]
        print x[-1]
        print '=='

        print 'x-est'
        print qmath_np.xadd(x[0], dx[0])
        print qmath_np.xadd(x[-1], dx[-1])
        #for x_, dx_ in zip(x, dx):
        #    print qmath_np.xadd(x_, dx_)
        print '=='
        #x = np.stack(x, axis=0) #(N,7) I think

        #np.save('x0.npy', x)
        #x = apply_delta_n(x, dx)
        #np.save('x.npy', x)
        #print 'x-est', x

        # TODO : implement online version, maybe adapt Sebastian Thrun's code related to online slam
        # where relationships regarding x_{i-1} can be folded into x_{i}

    def _build(self):
        print 'build begin ... '
        x0 = ['x','y','z','qx','qy','qz','qw']
        dx0 = [('d'+e) for e in x0[:6]]

        xi_s = symbols([e+'_i' for e in x0])
        xi = sm.Matrix(xi_s)

        xj_s = symbols([e+'_j' for e in x0])
        xj = sm.Matrix(xj_s)

        zij_s = symbols([e+'_ij_z' for e in x0])
        zij = sm.Matrix(zij_s)

        eij = err(xi,xj,zij)

        Aij = eij.jacobian(xi)
        Bij = eij.jacobian(xj)

        dxi_s = symbols([e+'_i' for e in dx0])
        dxi = sm.Matrix(dxi_s)

        Mi = apply_delta(xi, dxi)
        Mi = Mi.jacobian(dxi)
        Mi = Mi.subs({e:0 for e in dxi_s})

        dxj_s = symbols([e+'_j' for e in dx0])
        dxj = sm.Matrix(dxj_s)

        Mj = apply_delta(xj, dxj)
        Mj = Mj.jacobian(dxj)
        Mj = Mj.subs({e:0 for e in dxj_s})

        Aij_m = Aij * Mi
        #Aij_m.simplify()

        Bij_m = Bij * Mj
        #Bij_m.simplify()

        syms = (xi_s, xj_s, zij_s)
        print 'build complete'
        return syms, Aij_m, Bij_m, eij

def main():
    dx_p = 0.1 #
    dx_q = np.deg2rad(10.0)
    dz_p = 0.1
    dz_p = np.deg2rad(10.0)

    n_t = 100 # timesteps
    n_l = 4 # landmarks
    with np.errstate(invalid='raise'):

        max_nodes = n_t + n_l
        slam = GraphSlam3()
        zs, zs_gt, (p,q) = gen_data(n_t, n_l, 1e-1, 1e-1)
        slam.run(zs, max_nodes=max_nodes)
        np.save('zs_gt.npy', zs_gt)
        print 'p', p
        print 'q', q

        print 'zs_gt'
        for z in zs_gt:
            print z
        print '=='

        np.save('p_gt.npy', p)
        np.save('q_gt.npy', q)

if __name__ == "__main__":
    main()
