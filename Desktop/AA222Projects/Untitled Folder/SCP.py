import numpy as np
import gen_wind_field
import dynamics

# define initial variables
m, g, rho, s, Cd, wind_field = (
    1,
    9.8,
    1.225,
    1,
    0.05,
    gen_wind_field.WindField(
        [[0, 10], [0, 10]], [10, 10], [0, 0, 0], [1, 1, 1], [[2, 7]]
    ),
)

dyn = dynamics.Dynamics(m, g, rho, s, Cd, wind_field)
"""
Set up initial state and control parameters
state = x, y, z, V, gamma, psi
control = Cl, phi, T
"""
x, y, z, V, gamma, psi = 2, 3, 5, 5, np.pi / 36, np.pi / 9
Cl, phi, T = 0.8, np.pi / 18, 0.8
state = np.array([x, y, z, V, gamma, psi])
control = np.array([Cl, phi, T])
"""
Evaluate the current state dynamics and Jacobians
"""
fdot = dyn.f(state, control)
xdot, ydot, zdot, Vdot, gammadot, psidot = (
    fdot[0],
    fdot[1],
    fdot[2],
    fdot[3],
    fdot[4],
    fdot[5],
)

print(dyn.dfdx(state, control))
print(dyn.dfdu(state, control))
