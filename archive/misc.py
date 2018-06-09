import sympy as sm
R = sm.Matrix(sm.MatrixSymbol('R', 3, 3))
Rt = sm.Matrix(R.T)
print Rt
print Rt.jacobian(R)
