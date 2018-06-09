from tf import transformations as tx
import sympy as sm
import numpy as np

def T(q):
    return q[:3,:]

def Tinv(qv):
    x,y,z = qv[:,0]
    w = sm.sqrt(1.0 - qv.norm()**2)
    return sm.Matrix([x,y,z,w])

def qmul(q1, q0):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return sm.Matrix([[x1*w0 + y1*z0 - z1*y0 + w1*x0],
            [-x1*z0 + y1*w0 + z1*x0 + w1*y0],
            [x1*y0 - y1*x0 + z1*w0 + w1*z0],
            [-x1*x0 - y1*y0 - z1*z0 + w1*w0]])

def qinv(q):
    x, y, z, w = q[:,0]
    return sm.Matrix([x,y,z,-w])

def qxv(q, v, norm=True):
    s = v.norm()
    x,y,z = v[:,0]
    q_v = sm.Matrix([x,y,z,0])
    q_c = qinv(q)
    v = qmul(qmul(q, q_v),q_c)[:-1, :] # == q.v.q^{-1}
    v = np.multiply(v,s)
    return v

def apply_delta(x, dx):
    t, q = x[:3,:], x[3:,:]
    dt, dq_v = dx[:3,:], dx[3:,:]

    t_new = t + qxv(q,dt)#qmul(q, dt)
    print t_new
    q_new = qmul(q, Tinv(dq_v)) # TODO : or inverse????
    return t_new.col_join(q_new)

def err(xi, xj, zij):
    ti, qi = xi[:3,:], xi[3:,:]
    tj, qj = xj[:3,:], xj[3:,:]
    tij_z, qij_z = zij[:3,:], zij[3:,:]

    e_t = qxv(qinv(qi), tj - ti) - tij_z
    e_t.simplify()
    e_q = qmul(qmul(qinv(qi),qj),qinv(qij_z))
    e_q.simplify()
    return e_t.col_join(e_q[:3,:])
    #return e_t, e_q

#def qlog(q):
#    """ Quaternion Log """
#    va, ra = q[:3], q[-1]
#    uv = va / np.linalg.norm(va)
#    return uv * np.arccos(ra)
#
#def qvexp(qv):
#    """ Rotation-Vector Exponential """
#    ac = np.linalg.norm(qv, axis=-1)
#    ra = np.cos(ac)
#    va = (np.sin(ac)/ac) * qv
#    q = np.concatenate([va, [ra]], axis=-1)
#    return q / np.linalg.norm(q)

#def qlog(q):
#    s = q[:3, :].norm()
#    if(s == 0): # TODO : check EPS instead
#        return sm.Matrix([0,0,0])
#    else:
#        return q[:3,:] * (sm.acos(q[0] / s))
#
#def qvexp(qv):
#    ac = qv.norm()
#    if ac == 0: # TODO : check EPS instead
#        return sm.Matrix([0,0,0,1])
#    else:
#        ra = sm.cos(ac)
#        va = (sm.sin(ac)/ac) * qv
#        print 'va', va
#        return sm.Matrix([va[0], va[1], va[2], ra])

def symbols(*args):
    return sm.symbols(*args, real=True)

def main():
    x0 = ['x','y','z','qx','qy','qz','qw']
    dx0 = [('d'+e) for e in x0[:6]]

    xi_s = symbols([e+'_i' for e in x0])
    xi = sm.Matrix(xi_s)

    xj = symbols([e+'_j' for e in x0])
    xj = sm.Matrix(xj)

    zij = symbols([e+'_ij_z' for e in x0])
    zij = sm.Matrix(zij)

    #eij = err(xi,xj,zij)

    #Aij = eij.jacobian(xi)
    #Bij = eij.jacobian(xj)

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

    sm.pprint(Mi)
    sm.pprint(Mj)
    return

    print 'Mi'
    sm.pprint(Mi)

    print 'Mj'
    sm.pprint(Mj)

    Aij_m = Aij * Mi
    Aij_m.simplify()

    Bij_m = Bij * Mj
    Bij_m.simplify()

    print 'Aij_m'
    sm.pprint(Aij_m)

    print Aij_m.shape
    print eij.shape

    #x = sm.symbols('x,y,z,qx,qy,qz,qw')
    #x = sm.Matrix(x)
    #dx = sm.symbols('dx,dy,dz,dqx,dqy,dqz')
    #dx = sm.Matrix(dx)
    #J = apply_delta(x, dx).jacobian(dx) # M
    #J.simplify()
    #print 'J', J

    #ax = np.random.uniform(-1, 1, size=3)
    #ax /= np.linalg.norm(ax)
    #h = np.random.uniform(-np.pi, np.pi, size=1)
    #q = tx.quaternion_about_axis(h, ax)
    #
    #print q
    #q = sm.Matrix(q)
    #qq = qvexp(qlog(q)).normalized()

if __name__ == "__main__":
    main()
