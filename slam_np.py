from tf import transformations as tx
import numpy as np

def rq():
    ax = np.random.uniform(-1.0,1.0,size=3)
    ax /= np.linalg.norm(ax)
    h = np.random.uniform(-np.pi,np.pi)
    return tx.quaternion_about_axis(h,ax)

def rt():
    return np.random.uniform(-1.0, 1.0, size=3)

qx, qy, qz, qw = rq()
t = rt()
x,y,z = t

A = [
        [[0, -4*qy, -4*qz, 0], [2*qy, 2* qx, 2*qw, 2* qz], [2*qz, -2*qw, 2*qx, -2*qy]],
        [[2*qy, 2*qx, -2*qw, -2*qz], [-4*qx, 0, -4*qz, 0], [2*qw, 2*qz, 2*qy, 2*qx]],
        [[2*qz, 2*qw, 2*qx, 2*qy], [-2*qw, 2*qz, 2*qy, -2*qx], [-4*qx, -4*qy, 0, 0]]]
J2 = [[2*qy*y + 2*qz*z, -4*qy*x + 2*qx*y - 2*qw*z, -4*qz*x + 2*qw*y + 2*qx*z, 2*qz*y - 2*qy*z],
        [2*qy*x - 4*qx*y + 2*qw*z, 2*qx*x + 2*qz*z, -2*qw*x - 4*qz*y + 2*qy*z, -2*qz*x + 2*qx*z],
        [2*qz*x - 2*qw*y - 4*qx*z, 2*qw*x + 2*qz*y - 4*qy*z, 2*qx*x + 2*qy*y, 2*qy*x - 2*qx*y]]
J2 = np.asarray(J2)
A = np.asarray(A)
J = np.einsum('ijk,j -> ik', A, t)
print J
print J2
print A.shape

qx2, qy2, qz2, qw2 = rq()
qx3, qy3, qz3, qw3 = rq()
qx4, qy4, qz4, qw4 = rq()

dqq = [[-qw2*qw3-qx2*qx3-qy2*qy3-qz2*qz3,-qx3*qy2+qx2*qy3-qw3*qz2+qw2*qz3,qw3*qy2-qw2*qy3-qx3*qz2+qx2*qz3,qw3*qx2-qw2*qx3+qy3*qz2-qy2*qz3],[qx3*qy2-qx2*qy3+qw3*qz2-qw2*qz3,-qw2*qw3-qx2*qx3-qy2*qy3-qz2*qz3,-qw3*qx2+qw2*qx3-qy3*qz2+qy2*qz3,qw3*qy2-qw2*qy3-qx3*qz2+qx2*qz3],[-qw3*qy2+qw2*qy3+qx3*qz2-qx2*qz3,qw3*qx2-qw2*qx3+qy3*qz2-qy2*qz3,-qw2*qw3-qx2*qx3-qy2*qy3-qz2*qz3,qx3*qy2-qx2*qy3+qw3*qz2-qw2*qz3],[qw3*qx2-qw2*qx3+qy3*qz2-qy2*qz3,qw3*qy2-qw2*qy3-qx3*qz2+qx2*qz3,qx3*qy2-qx2*qy3+qw3*qz2-qw2*qz3,qw2*qw3+qx2*qx3+qy2*qy3+qz2*qz3]]
dqq = np.asarray(dqq)
print dqq
