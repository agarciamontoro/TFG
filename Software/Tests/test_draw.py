import unittest
from hypothesis import given, assume
from hypothesis.strategies import floats, tuples, sampled_from, integers

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


def cart2spher(points):
    """Converts an array of points in cartesian coordinates to spherical

    Args:
        points (NumPy array): A NumPy 2D array, where every row contains (x,
            y, z), the cartesian coordinates of a point.
    """
    # Define the new array
    sphericalPoints = np.zeros(points.shape)

    xy = points[:, 0]**2 + points[:, 1]**2

    sphericalPoints[:, 0] = np.sqrt(xy + points[:, 2]**2)
    sphericalPoints[:, 1] = np.arctan2(np.sqrt(xy), points[:, 2])
    sphericalPoints[:, 2] = np.arctan2(points[:, 1], points[:, 0])

    return sphericalPoints


def spher2cart(points):
    """Converts an array of points in spherical coordinates to cartesian

    Args:
        points (NumPy array): A NumPy 2D array, where every row contains (r,
            theta, phi), the spherical coordinates of a point.
    """
    # Define the new array
    cartesianPoints = np.zeros(points.shape)

    # Compute r*sin(phi)
    rSinTheta = points[:, 0] * np.sin(points[:, 1])

    # Compute x, y, z coordinates and store it in the new array.
    # The coordinate transformation done is as follows:
    #   x = r * sin(theta) * cos(phi)
    #   y = r * sin(theta) * sin(phi)
    #   z = r * cos(theta)
    cartesianPoints[:, 0] = rSinTheta * np.cos(points[:, 2])
    cartesianPoints[:, 1] = rSinTheta * np.sin(points[:, 2])
    cartesianPoints[:, 2] = points[:, 0] * np.cos(points[:, 1])

    return cartesianPoints

#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# X, Y, Z = axes3d.get_test_data(0.05)
#
# spher = cart2spher(np.column_stack((X, Y, Z)))
# cart  = spher2cart(spher)
#
# # X = cart[:, 0]
# # Y = cart[:, 1]
# # Z = cart[:, 2]
#
# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
#
# plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# import numpy as np
# from matplotlib import cm
# from matplotlib import pyplot as plt
# step = 0.04
# maxval = 1.0
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # create supporting points in polar coordinates
# r = np.linspace(0, 1.25, 50)
# p = np.linspace(0, 2*np.pi, 50)
# R, P = np.meshgrid(r, p)
# # transform them to cartesian system
# X, Y = R*np.cos(P), R*np.sin(P)
#
# Z = ((R**2 - 1)**2)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
# ax.set_zlim3d(0, 1)
# ax.set_xlabel(r'$\phi_\mathrm{real}$')
# ax.set_ylabel(r'$\phi_\mathrm{im}$')
# ax.set_zlabel(r'$V(\phi)$')
# plt.show()

class TestCoordinateSystem(unittest.TestCase):
    @given(floats(), floats(), floats())
    def testCartesian(self, x, y, z):
        assume(abs(x) + abs(y) + abs(z) > 1e-10)
        assume(abs(x) < 1e10)
        assume(abs(y) < 1e10)
        assume(abs(z) < 1e10)

        actual = np.column_stack((x, y, z))

        expected = spher2cart(cart2spher(actual))

        np.testing.assert_almost_equal(actual, expected, decimal=3)

    @given(floats(), floats(), floats())
    def testSpherical(self, r, theta, phi):
        assume(abs(r) < 1e10)
        assume(abs(theta) < 1e10)
        assume(abs(phi) < 1e10)
        assume(r > 1)
        # assume(0 <= theta < 2*np.pi)
        # assume(0 <= phi <= np.pi)

        actual = np.column_stack((r, theta, phi))

        expected = cart2spher(spher2cart(actual))

        np.testing.assert_almost_equal(actual, expected, decimal=3)


# unittest.main()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# theta = np.linspace(0, 2 * np.pi, 100)
# phi = np.linspace(0, 2 * np.pi, 100)
# r = np.repeat(1, 100)
#
# cart = spher2cart(np.column_stack((r, theta, phi)))
# x = cart[:, 0]
# y = cart[:, 1]
# z = cart[:, 2]
#
# ax.scatter(x, y, z)
# ax.legend()
#
# plt.show()

# r = 1.
# theta = 0.
# phi = 0.0015
#
# spher = np.column_stack((r, theta, phi))
# print(spher)
# print(spher2cart(spher))
# print(cart2spher(spher2cart(spher)))

x = np.array(range(-100, 101))
y = np.array(range(-100, 101))
d = 10000

theta = np.arctan2(y/np.sqrt(d**2-np.square(x)))
phi   = np.arctan2(np.divide(x, d))
