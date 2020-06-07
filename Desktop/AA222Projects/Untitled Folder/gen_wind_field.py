import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


class WindField:
    def __init__(self, bounds, resolution, mu, sigma, thermal_list):
        """
        generate wind field. Create bi-variate spline objects to represent wind in each direction. Coordinate system NED
        :param bounds: list of two lists, [[x lb, x ub], [y lb, y ub]]. xy bounds on the wind field
        :param resolution: list of int, number of points in x and y direction
        :param mu: list of float, average wind speed in xyz directions
        :param sigma: list of float, standard deviation of wind speed in xyz directions
        :param thermal_list: list of list, coordinates of thermal location
        """
        self.wind = [None, None, None]
        self.bounds = bounds
        x = np.linspace(bounds[0][0], bounds[0][1], resolution[0])
        y = np.linspace(bounds[1][0], bounds[1][1], resolution[1])
        for i in range(3):
            z = np.random.normal(mu[i], sigma[i], resolution)
            if thermal_list and i == 2:
                for coord in thermal_list:
                    x_mesh, y_mesh = np.meshgrid(x, y)
                    r_sq = (x_mesh - coord[0]) ** 2 + (y_mesh - coord[1]) ** 2
                    thermal_wind = np.exp(-r_sq) * sigma[i] * 10
                    # subtract, because we are in NED frame
                    z = z - np.transpose(thermal_wind)
            self.wind[i] = RectBivariateSpline(x, y, z)
        return

    def get_wind_vel(self, coord):
        """
        get wind speed at coordinate(s)
        :param coord: np array; if 1d array, will be treated as single coordinate; if 2d array, must have dimension n*2,
         and each row contains a coordinate
        :return: if coord is 1d, returns a 1d array containing wind velocity in xyz direction at that point;
        if coord is 2d and has dimension n*2, returns a n*3 array containing wind velocity at queried coordinates
        """
        if len(np.shape(coord)) == 1:
            vel = np.zeros(3)
            for i in range(3):
                vel[i] = self.wind[i].ev(coord[0], coord[1])
        else:
            vel = np.zeros([np.shape(coord)[0], 3])
            for i in range(3):
                vel[:, i] = self.wind[i].ev(coord[:, 0], coord[:, 1])
        return vel

    def get_wind_grad(self, coord):
        """
        get gradient of wind speed at coordinate(s)
        :param coord: np array; if 1d array, will be treated as single coordinate; if 2d array, must have dimension n*2,
         and each row contains a coordinate
        :return: if coord is 1d, returns a 3*2 array, containing dWi/dxj; if coord is 2d n*2, returns n*3*2 array,
        where first index goes over the coordinates, second index is over wind direction, and the third index is xy
        """
        if len(np.shape(coord)) == 1:
            grad = np.zeros([3, 2])
            for i in range(3):
                grad[i, 0] = self.wind[i].ev(coord[0], coord[1], 1, 0)
                grad[i, 1] = self.wind[i].ev(coord[0], coord[1], 0, 1)
        else:
            grad = np.zeros([np.shape(coord)[0], 3, 2])
            for i in range(3):
                grad[:, i, 0] = self.wind[i].ev(coord[:, 0], coord[:, 1], 1, 0)
                grad[:, i, 1] = self.wind[i].ev(coord[:, 0], coord[:, 1], 0, 1)
        return grad

    def plot_wind(self, plot_flag):
        """
        generate contour plots for wind speed in xyz directions
        :param plot_flag: boolean array of length 3, whether to plot the wind in each direction
        :return:
        """
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], 50)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], 50)
        vel = np.zeros([len(x), len(y), 3])
        for i in range(len(x)):
            for j in range(len(y)):
                vel[i, j, :] = self.get_wind_vel([x[i], y[j]])
        for i in range(3):
            if plot_flag[i]:
                plt.contour(x, y, vel[:, :, i], 10)
                plt.title("direction {0}".format(i))
                plt.colorbar()
                plt.show()
        return


# def test():
#     test_wind = WindField([[0, 10], [0, 10]], [10, 10], [0, 0, 0], [1, 1, 1], [[2, 7]])
#     test_wind.plot_wind([1, 1, 1])
#     return


# test()
