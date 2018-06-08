import numpy as np
import sympy as sm
from tf import transformations as tx

from gen_data import gen_data
from qmath import *

eps = np.finfo(float).eps

def symbols(*args):
    return sm.symbols(*args, real=True)

def qxv(q, v):

    s = np.linalg.norm(v)
    if s <= eps:
        # don't bother.
        return 0 * v

    # else ...
    v = np.divide(v, s) # make unit

    q_v = np.concatenate((v,[0]), axis=-1)
    q_c = tx.quaternion_conjugate(q)
    v = tx.quaternion_multiply(
            tx.quaternion_multiply(q, q_v),
            q_c)# == q.v.q^{-1}

    v = np.multiply(v[:-1],s)
    return v

def xzadd(x, z):
    xp,xq = x[:3], x[3:]
    zp,zq = z[:3], z[3:]
    p = qxv(xq, zp)
    q = tx.quaternion_multiply(xq, zq)
    return np.concatenate((p,q), axis=-1)

class GraphSlam3(object):
    def __init__(self):
        syms, Aij, Bij, eij = self._build()
        self._syms = syms
        self._Aij = Aij
        self._Bij = Bij
        self._eij = eij

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
        values = (n[i0], n[i1], x)
        sargs = {}
        for (ks, vs) in zip(self._syms, values):
            sargs.update({k:v for k,v in zip(ks,vs)})
        Aij = self._Aij.subs(sargs)
        Bij = self._Bij.subs(sargs)
        eij = self._eij.subs(sargs)

        # convert to np
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
                self._nodes[z0] = z

            if z1 not in self._nodes:
                # add initial guesses?
                self._nodes[z1] = xzadd(self._nodes[z0], z)

            Aij, Bij, eij = self.add_edge(z, z0, z1)

            try:
                H[z0][z0] += np.matmul(Aij.T, Aij)
                H[z0][z1] += np.matmul(Aij.T, Bij)
                H[z1][z0] += np.matmul(Bij.T, Aij)
                H[z1][z1] += np.matmul(Bij.T, Bij)
                b[z0]     += np.matmul(Aij.T, eij)
                b[z1]     += np.matmul(Bij.T, eij)
            except Exception:
                print z0, z1
        H[0][0] += np.eye(6)

        H = np.block(H)
        #b = np.block(b)
        b = np.concatenate(b, axis=0)

        dx = np.matmul(np.linalg.pinv(H), -b)
        np.save('dx.npy', dx)

    def _build(self):
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
        Aij_m.simplify()

        Bij_m = Bij * Mj
        Bij_m.simplify()

        syms = (xi_s, xj_s, zij_s)
        return syms, Aij_m, Bij_m, eij

def main():
    n_t = 5
    n_l = 4

    max_nodes = n_t + n_l
    slam = GraphSlam3()
    zs = gen_data(n_t, n_l)
    slam.run(zs, max_nodes=max_nodes)

if __name__ == "__main__":
    main()
