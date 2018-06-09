import numpy as np
import sympy as sm
from tf import transformations as tx
from sympy import Function
from sympy.core.singleton import S
from sympy.functions.elementary.trigonometric import _pi_coeff

class unorm(Function):
    @classmethod
    def eval(cls, x):
        if x is S.Zero:
            return S.One
        else:
            return x

    def _eval_is_real(self):
        return self.args[0].is_real

    def fdiff(argindex=1):
        print self.args[0]
        return S.One

def qlog(q):
    """ Quaternion Log """
    va, ra = q[:3,:], q[-1, 0]
    n = unorm(va.norm())
    s = sm.acos(ra)/n
    res = va * s
    return res


def qvexp(qv):
    """ Rotation-Vector Exponential """
    ac = qv.norm()
    va = sm.sinc(ac) * qv
    ra = sm.cos(ac)
    return sm.Matrix([va[0],va[1],va[2],ra])
    #return va.col_join(ra)

def T(q):
    return q[:3,:]

def Tinv(qv):
    x,y,z = qv[:,0]
    w = sm.sqrt(1.0 - qv.norm()**2)
    return sm.Matrix([x,y,z,w])

#def T(q):
#    return qlog(q)
#
#def Tinv(qv):
#    return qvexp(qv)

def qmul(q1, q0):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return sm.Matrix([[x1*w0 + y1*z0 - z1*y0 + w1*x0],
            [-x1*z0 + y1*w0 + z1*x0 + w1*y0],
            [x1*y0 - y1*x0 + z1*w0 + w1*z0],
            [-x1*x0 - y1*y0 - z1*z0 + w1*w0]])

def q2R(q):
    qx, qy, qz, qw = q
    qx2,qy2,qz2,qw2 = qx*qx, qy*qy, qz*qz, qw*qw
    M = [[1 - 2*qy2 - 2*qz2,	2*qx*qy - 2*qz*qw,	2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,	1 - 2*qx2 - 2*qz2,	2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,	2*qy*qz + 2*qx*qw,	1 - 2*qx2 - 2*qy2]]
    return sm.Matrix(M)

def qinv(q):
    x, y, z, w = q[:,0]
    return sm.Matrix([x,y,z,-w]).normalized()

def qxv(q, v):
    return q2R(q)*v
    #s = v.norm()
    #x,y,z = v[:,0]
    #q_v = sm.Matrix([x,y,z,0])
    #q_c = qinv(q)
    #v = qmul(qmul(q, q_v),q_c)[:-1, :] # == q.v.q^{-1}
    #v = v * s
    #return v

def apply_delta(x, dx):
    t, q = x[:3,:], x[3:,:]
    dt, dq_v = dx[:3,:], dx[3:,:]

    t_new = t + dt#qmul(q, dt)
    q_new = qmul(Tinv(dq_v), q) # TODO : or inverse????
    return t_new.col_join(q_new)

def err(xi, xj, zij):
    ti, qi = xi[:3,:], xi[3:,:]
    tj, qj = xj[:3,:], xj[3:,:]
    tij_z, qij_z = zij[:3,:], zij[3:,:]

    e_t = qxv(qinv(qi), tj - ti) - tij_z
    e_t.simplify()
    e_q = qmul(qmul(qinv(qi),qj),qinv(qij_z))
    e_q.simplify()
    res = e_t.col_join(T(e_q))
    return res

def symbols(*args):
    s = sm.symbols(*args, real=True)
    return sm.Matrix(s), s

def err_test(xi, xj, zij):
    ti, qi = xi
    tj, qj = xj
    tij_z, qij_z = zij

    qi_i = qinv(qi)

    #e_t = qxv(qi_i, tj - ti)# - tij_z
    e_t = qxv(qi_i, tj)# - qxv(qi_i, ti)# - tij_z

    e_t.simplify()

    e_q = qmul(qmul(qi_i,qj),qinv(qij_z))
    e_q = T(e_q)
    e_q.simplify()
    return e_t, e_q

def rq():
    ax = np.random.uniform(-1.0,1.0,size=3)
    ax /= np.linalg.norm(ax)
    h = np.random.uniform(-np.pi,np.pi)
    return tx.quaternion_about_axis(h,ax)

def rt():
    return np.random.uniform(-1.0, 1.0, size=3)

def main():
    t0 = ['x','y','z']
    q0 = ['qx','qy','qz','qw']
    dt = sm.symbols('dx, dy, dz')

    ti, ti_s = symbols([e + '_i' for e in t0])
    qi, qi_s = symbols([e + '_i' for e in q0])
    tj, tj_s = symbols([e + '_j' for e in t0])
    qj, qj_s = symbols([e + '_j' for e in q0])
    tij_z, tij_z_s = symbols([e + '_ij_z' for e in t0])
    qij_z, qij_z_s = symbols([e + '_ij_z' for e in q0])

    e_t, e_q = err_test([ti,qi], [tj,qj], [tij_z, qij_z])
    J = e_t.jacobian(qi) # == equivalent to -R(q_i).T

    qiv = rq()
    print qiv
    tjv = rt()
    sargs = {k:v for k,v in zip(qi_s, qiv)}
    #sargs.update({k:v for k,v in zip(tj_s, tjv)})

    print 'J1 : Raw'
    sm.pprint(J.subs(sargs))
    #J = J.subs({
    #    qi.norm() : 1.0,
    #    qj.norm() : 1.0,
    #    ti.norm() : 'n_ti',
    #    tj.norm() : 'n_tj'})
    #print J.shape
    #J.simplify()
    #sm.pprint(J)

    # careful assembly  ...
    R = q2R(qi_s)
    mR = sm.Matrix(sm.flatten(R.T))

    Jx = mR.jacobian([qi[0]])
    Jy = mR.jacobian([qi[1]])
    Jz = mR.jacobian([qi[2]])
    Jw = mR.jacobian([qi[3]])

    Jx = Jx.reshape(3,3) * tj
    Jy = Jy.reshape(3,3) * tj
    Jz = Jz.reshape(3,3) * tj
    Jw = Jw.reshape(3,3) * tj

    J = Jx.row_join(Jy).row_join(Jz).row_join(Jw)
    print J.shape

    #J = .jacobian(qi) # 9x4 with (RixRj, Q)
    #J = J.reshape(3, 3*4)
    #J = (J.T * tj).T
    #J = J.reshape(3,4)
    #print J.shape
    #J = J.T # 4x9
    #J = J.reshape(4*3,3)
    #J = (J * tj).T # 4*3, 1)
    #J = J.reshape(3,4)
    #print J.shape
    
    print 'J2 : Smart'
    sm.pprint(J.subs(sargs))
    #J = J.subs({
    #    qi.norm() : 1.0,
    #    qj.norm() : 1.0,
    #    ti.norm() : 'n_ti',
    #    tj.norm() : 'n_tj'})
    #J.simplify()
    #print 'J2 : Smart?'
    #sm.pprint(J)
    
    #J.simplify()

    #print 'J1'
    #sm.pprint(J)

    #J2 = qxv(qinv(qi), ti).jacobian(ti)
    #J2.simplify()

    #print 'J1'
    #sm.pprint(R.subs({k:v for k,v in zip(qi_s, qiv)}))
    #sm.pprint(J.subs({k:v for k,v in zip(qi_s, qiv)}))

    #print 'J2'
    #sm.pprint(J2)

    #for i in range(3):
    #    J = J.subs(ti[i]-tj[i], dt[i])
    #J = J.subs(dt[0]**2 + dt[1]**2 + dt[2]**2, 'dt2')
    #J.simplify()
    #print J

if __name__ == "__main__":
    main()
