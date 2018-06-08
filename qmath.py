import sympy as sm

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
    v = v * s
    return v

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
    return e_t.col_join(e_q[:3,:])

