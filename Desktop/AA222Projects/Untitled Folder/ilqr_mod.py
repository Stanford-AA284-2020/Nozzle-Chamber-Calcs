import numpy as np

class InitialTrajectory:
    def __init__(self, vehicle_dyn, x0,x_goal, t_goal, t_step):
        """
        initialize parameters
        :param vehicle_dyn: Dynamics object, vehicle dynamics
        :param cost_func: Cost object, cost function, includes penalty terms
        :param x0: 1d array, initial state
        :param t_goal: float, time to reach goal
        :param t_step: float, time step used
        """
        self.dynamics = vehicle_dyn
        self.x0 = x0
        self.t_goal = t_goal
        self.x_goal = x_goal
        self.N = int(t_goal / float(t_step))
        self.t_array, self.t_step = np.linspace(0, t_goal, self.N, retstep=True)
        return

    def init_control(self):
        """
        initialize control and state trajectory. based on straight level un-accelerated flight from initial state to
        goal when there is no wind
        :return: u_ref, N*m array, each row contains the control for that time step
                x_ref, N*n array, each row contains the state for that time step
        """
        V = np.linalg.norm(self.x_goal - self.x0) / float(self.t_goal)
        q = 0.5 * self.dynamics.rho * V**2
        CL = self.dynamics.m * self.dynamics.g / (q * self.dynamics.S) + 0.001
        phi = 0
        T = self.dynamics.Cd * q * self.dynamics.S
        u_ref = np.tile([CL, phi, T], [self.N, 1])
        x_ref = np.tile(self.x0, [self.N, 1])
        for i in range(self.N - 1):
            x_ref[i + 1, :] = x_ref[i, :] + self.dynamics.f(x_ref[i, :], u_ref[i, :]) * self.t_step
        return u_ref, x_ref
