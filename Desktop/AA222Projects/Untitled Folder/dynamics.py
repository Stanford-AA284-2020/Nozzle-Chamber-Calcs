import numpy as np
import gen_wind_field


class Dynamics:
    def __init__(self, m, g, rho, s, Cd, wind_field):
        """
        initialize parameters for vehicle dynamics, all units in SI and all angles in radian
        :param m: float, mass of vehicle, kg
        :param g: float, gravitional acceleration, m/s2
        :param rho: float, density of air, kg/m3
        :param s: float, reference area, m2
        :param Cd: float, drag coefficient
        :param wind_field: WindField object, detail see gen_wind_field.py
        """
        self.m = m
        self.g = g
        self.rho = rho
        self.S = s
        self.Cd = Cd
        self.wind = wind_field
        return

    def f(self, state, control):
        """
        calculate vehicle dynamics f(x, u) = \dot{x}
        :param state: 1d array of floats, vehicle state x, y, z, V, gamma (flight path angle), psi (heading angle),
        all angles in radian
        :param control: 1d array of floats, control, CL, phi (bank angle, radian), T (thrust, N)
        :return: 1d array, f(x, u) = \dot{x}
        """
        x, y, z, V, gamma, psi = state
        Cl, phi, T = control
        Wx, Wy, Wz = self.wind.get_wind_vel(np.array([x, y, z]))
        D = self.Cd * 0.5 * self.rho * V ** 2 * self.S
        L = Cl * 0.5 * self.rho * V ** 2 * self.S
        f = np.zeros(6)
        f[0] = V * np.cos(gamma) * np.cos(psi) + Wx
        f[1] = V * np.cos(gamma) * np.sin(psi) + Wy
        f[2] = -V * np.sin(gamma) + Wz
        dWdx = self.wind.get_wind_grad(np.array([x, y, z]))
        dWdt = np.dot(dWdx, f[0:2])
        f[3] = (
            T
            - D
            - self.m * self.g * np.sin(gamma)
            - self.m * dWdt[0] * np.cos(gamma) * np.cos(psi)
            - self.m * dWdt[1] * np.cos(gamma) * np.sin(psi)
            + self.m * dWdt[2] * np.sin(gamma)
        )
        f[4] = (1.0 / (self.m * V)) * (
            L * np.cos(phi)
            - self.m * self.g * np.cos(gamma)
            + self.m * dWdt[0] * np.sin(gamma) * np.cos(psi)
            + self.m * dWdt[1] * np.sin(gamma) * np.sin(psi)
            + self.m * dWdt[2] * np.cos(gamma)
        )
        f[5] = (1.0 / (self.m * V * np.cos(gamma))) * (
            L * np.sin(phi)
            + self.m * dWdt[0] * np.sin(psi)
            - self.m * dWdt[1] * np.cos(psi)
        )
        return f

    def dfdx(self, state, control):
        """
        calculate derivative of f with respect to vehicle state using finite difference
        :param state: 1d array of floats, vehicle state x, y, z, V, gamma (flight path angle), psi (heading angle),
        all angles in radian
        :param control: 1d array of floats, control, CL, phi (bank angle, radian), T (thrust, N)
        :return: dfdx, 6*6 array of floats, dfi/dxj
        """
        epsilon = 1e-6
        dfdx = np.zeros([6, 6])
        f0 = self.f(state, control)
        for i in range(6):
            state_perturbed = np.array(state)
            state_perturbed[i] += epsilon
            f_perturbed = self.f(state_perturbed, control)
            dfdx[:, i] = (f_perturbed - f0) / epsilon
        return dfdx

    def dfdu(self, state, control):
        """
        calculate derivative of f with respect to controls
        :param state: 1d array of floats, vehicle state x, y, z, V, gamma (flight path angle), psi (heading angle),
        all angles in radian
        :param control: 1d array of floats, control, CL, phi (bank angle, radian), T (thrust, N)
        :return: dfdu, 6*3 array of floats, dfi/duj
        """
        V = state[3]
        Cl = control[0]
        phi = control[1]
        dfdu = np.zeros([6, 3])
        dfdu[3, 2] = 1
        dfdu[4, 0] = 0.5 * self.rho * V ** 2 * self.S * np.cos(phi)
        dfdu[4, 1] = -Cl * 0.5 * self.rho * V ** 2 * self.S * np.sin(phi)
        dfdu[5, 0] = 0.5 * self.rho * V ** 2 * self.S * np.sin(phi)
        dfdu[5, 1] = Cl * 0.5 * self.rho * V ** 2 * np.cos(phi)
        return dfdu


# def test():
# dyn = Dynamics(
#     1,
#     9.8,
#     1.225,
#     1,
#     0.05,
#     gen_wind_field.WindField(
#         [[0, 10], [0, 10]], [10, 10], [0, 0, 0], [1, 1, 1], [[2, 7]]
#     ),
# )


#     state = np.array([2, 3, 5, 5, np.pi / 36, np.pi / 9])
#     control = np.array([0.8, np.pi / 18, 0.8])
#     print(dyn.f(state, control))
#     print(dyn.dfdx(state, control))
#     print(dyn.dfdu(state, control))
#     return


# test()
