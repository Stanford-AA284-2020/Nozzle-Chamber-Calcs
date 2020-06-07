# cost
import numpy as np


class Cost:
    def __init__(self, rho, xlb, xub, ulb, uub, Qn, Qk, x_goal):
        """
        initialize parameters for cost function, which consists of power consumption, penalty for deviation from goal,
        and penalty for violating the bounds on state and control
        :param rho: float, penalty factor
        :param xlb: 1d array, lower bounds on state
        :param xub: 1d array, upper bounds on state
        :param ulb: 1d array, lower bounds on control
        :param uub: 1d array, upper bounds on control
        :param Qn: 2d matrix, penalty for final state, symmetric
        """
        self.rho = rho
        self.x_lb = xlb
        self.x_ub = xub
        self.u_lb = ulb
        self.u_ub = uub
        self.Qn = Qn
        self.Qk = Qk
        self.x_goal = x_goal
        self.m = np.size(ulb)
        self.n = np.size(xlb)
        return

    def c(self, state, control):
        """
        stage wise cost function
        :param state: 1d array of floats, vehicle state x, y, z, V, gamma (flight path angle), psi (heading angle),
        all angles in radian
        :param control: 1d array of floats, control, CL, phi (bank angle, radian), T (thrust, N)
        :return: c, float, stage wise cost
        """
        # energy consumption
        V = state[3]
        T = control[2]
        # cost = V * T
        cost = 0.5 * np.dot(control, np.dot(self.Qk, control))
        # penalty for constraints
        penalty = 0
        temp = np.maximum(control - self.u_ub, 0)
        penalty += np.dot(temp, temp)
        temp = np.maximum(self.u_lb - control, 0)
        penalty += np.dot(temp, temp)
        temp = np.maximum(state - self.x_ub, 0)
        penalty += np.dot(temp, temp)
        temp = np.maximum(self.x_lb - state, 0)
        penalty += np.dot(temp, temp)
        cost += self.rho * penalty
        return cost

    def cx(self, state, control):
        """
        calculate derivative of the cost with respect to state
        :param state: 1d array of floats, vehicle state
        :param control: 1d array of floats, control
        :return: dcdx, 1d array, derivative of the cost with respect to state
        """
        dcdx = np.zeros(self.n)
        for i in range(self.n):
            dcdx[i] = (
                max(2 * (state[i] - self.x_ub[i]), 0)
                + min(2 * (state[i] - self.x_lb[i]), 0)
            ) * self.rho
        # dcdx[3] += control[2]
        return dcdx

    def cu(self, state, control):
        """
        calculate derivative of cost with respect to control
        :param state: 1d array of floats, vehicle state
        :param control: 1d array of floats, control
        :return: dcdu, 1d array, derivative of the cost with respect to control
        """
        dcdu = np.dot(self.Qk, control)
        for i in range(self.m):
            dcdu[i] += (
                max(2 * (control[i] - self.u_ub[i]), 0)
                + min(2 * (control[i] - self.u_lb[i]), 0)
            ) * self.rho
        # dcdu[2] +=state[3]
        return dcdu

    def cxx(self, state, control):
        """
        calculate double derivative of cost with respect to state
        :param state: 1d array of floats, vehicle state
        :param control: 1d array of floats, control
        :return: dcdxx, n*n matrix
        """
        dcdxx = np.zeros([self.n, self.n])
        for i in range(self.n):
            if state[i] > self.x_ub[i] or state[i] < self.x_lb[i]:
                dcdxx[i, i] = 2
        return dcdxx

    def cuu(self, state, control):
        """
        calculate double derivatives of cost with respect to control
        :param state: 1d array of floats, vehicle state
        :param control: 1d array of floats, control
        :return: dcduu, m*m matrix
        """
        dcduu = self.Qk
        for i in range(self.m):
            if control[i] > self.u_ub[i] or control[i] < self.u_lb[i]:
                dcduu[i, i] += 2
        return dcduu

    def cux(self, state, control):
        """
        calculate derivative of cost with respect to control and state
        :param state: 1d array, vehicle state
        :param control: 1d array, control
        :return: dcdux, m*n array
        """
        dcdux = np.zeros([self.m, self.n])
        # dcdux[2, 3] = 1
        return dcdux

    def qn_matrix(self, state, control):
        """
        Qn matrix for final cost
        :return: Qn, n*n array
        """
        return self.Qn

    def qn_vector(self, state, control):
        """
        qn vector for final cost
        :return: 1d array
        """
        return -2 * np.dot(self.x_goal, self.Qn)

    def qn_scalar(self, state, control):
        """
        qn scalar for final cost
        :return: float
        """
        return np.dot(self.x_goal, np.dot(self.Qn, self.x_goal))


f_dot = dyn.f(state, control)
x_dot, y_dot, z_dot, V_dot, gamma_dot, psi_dot = (
    f_dot[0],
    f_dot[1],
    f_dot[2],
    f_dot[3],
    f_dot[4],
    f_dot[5],
)

