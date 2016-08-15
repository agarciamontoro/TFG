import unittest
import sys
import numpy as np
from hypothesis import given, assume
from hypothesis.strategies import floats, tuples, sampled_from, integers
from scipy.integrate import ode
from scipy.special import airy
from math import sqrt, sin, cos
sys.path.append('../RK4')
from rk4 import RK4Solver


# class TestRK4_HarmonicOscillator(unittest.TestCase):
#     """Test suite for RK4 solver <> y'' = -ay"""
#     def setUp(self):
#         self.dx = 0.01
#
#     @given(floats(), tuples(floats(), floats()), floats(max_value=0.),
#            floats(0.01, 9.0))
#     def testAnalyticalNegative(self, x0, y0, constant, intervalSize):
#         assume(constant != 0.)
#
#         RK4_functions = ["y[1]",
#                          "-(%.50f)*y[0]" % constant]
#
#         omega = sqrt(-constant)
#
#         def analyticalSolution(x0):
#             argument = omega*x0
#             return [y0[0]*cos(argument) + (y0[1]/omega)*sin(argument),
#                     y0[1]*cos(argument) - omega*y0[0]*sin(argument)]
#
#         # Own solver build (force double precision)
#         RK4_y0 = np.array([[y0]]).astype(np.float64)
#         RK4_solver = RK4Solver(x0, RK4_y0, self.dx, RK4_functions,
#                                debug=False)
#
#         # Evolution of the system
#         RK4_solver_y = RK4_solver.solve(RK4_solver.x0 + intervalSize)[0, 0, :]
#         analytical_y = analyticalSolution(RK4_solver.x0)
#
#         # Test that both data are almost the same, up to 3 decimal places
#         np.testing.assert_almost_equal(RK4_solver_y, analytical_y, decimal=3)
#
#     @given(floats(), tuples(floats(), floats()), floats(max_value=0.),
#            floats(0.01, 9.0))
#     def testNumericalNegative(self, x0, y0, constant, intervalSize):
#         assume(abs(constant) > 1)
#         assume(1e-5 < abs(y0[0]) < 1e5)
#         assume(1e-5 < abs(y0[1]) < 1e5)
#
#         RK4_functions = ["y[1]",
#                          "-(%.50f)*y[0]" % constant]
#
#         def SCI_functions(x, y, constant):
#             return [y[1], -constant*y[0]]
#
#         # SciPy solver build
#         SCI_solver = ode(SCI_functions).set_integrator('dopri5',
#                                                        first_step=self.dx)
#         SCI_solver.set_initial_value(y0, x0).set_f_params(constant)
#
#         # Own solver build (force double precision)
#         RK4_y0 = np.array([[y0]]).astype(np.float64)
#         RK4_solver = RK4Solver(x0, RK4_y0, self.dx, RK4_functions,
#                                debug=True)
#
#         # Evolution of the system
#         RK4_solver_y = RK4_solver.solve(RK4_solver.x0 + intervalSize)[0, 0, :]
#         SCI_solver.integrate(RK4_solver.x0)
#
#         # Test that both data are almost the same, up to 4 decimal places
#         np.testing.assert_almost_equal(RK4_solver_y, SCI_solver.y, decimal=4)


class TestRK4_Airy(unittest.TestCase):
    """Test suite for RK4 solver <> Airy function"""
    def setUp(self):
        self.dx = 0.001

        # Definition of system for own solver
        self.RK4_functions = ["y[1]", "x*y[0]"]

    @given(floats(-10., 0.), tuples(floats(-10., 10.), floats(-10., 10.)),
           floats(0.01, 9.0))
    def testAiryNumerical(self, x0, y0, intervalSize):
        "Tests Airy ODE y'' = xy against SciPy solver"

        # Definition of system for SciPy solver
        def SCI_functions(x, y):
            return [y[1], x*y[0]]

        # Own solver build
        RK4_y0 = np.array([[y0]]).astype(np.float64)
        RK4_solver = RK4Solver(x0, RK4_y0, self.dx, self.RK4_functions)

        # SciPy solver build
        SCI_solver = ode(SCI_functions).set_integrator('dopri5',
                                                       first_step=self.dx)

        SCI_solver.set_initial_value(y0, x0)

        # Evolution of the system with both solvers
        RK4_solver_y = RK4_solver.solve(RK4_solver.x0+intervalSize)[0, 0, :]
        SCI_solver.integrate(RK4_solver.x0)

        # Test that both data are almost the same, up to 5 decimal places
        np.testing.assert_almost_equal(RK4_solver_y, SCI_solver.y, decimal=5)


    @given(floats(max_value=0.), floats(0.01, 9.0))
    def testAiryAnalytical(self, x0, intervalSize):
        "Tests Airy ODE y'' = xy against analytical solution"

        # Assumptions: the interval does not exceeds 0 and the initial
        # condition time is not too big
        assume(x0 + intervalSize < 0.)
        assume(abs(x0) < 1e4)

        # Initial conditions taken from the airy function at the initial time
        Ai, Aip, Bi, Bip = airy(x0)
        y0 = [Ai, Aip]

        # Own solver build (force double precision)
        RK4_y0 = np.array([[y0]]).astype(np.float64)
        RK4_solver = RK4Solver(x0, RK4_y0, self.dx, self.RK4_functions)

        # Evolution of the system
        RK4_solver_y = RK4_solver.solve(RK4_solver.x0 + intervalSize)[0, 0, :]
        Ai, Aip, Bi, Bip = airy(RK4_solver.x0)

        # Test that both data are almost the same, up to 4 decimal places
        np.testing.assert_almost_equal(RK4_solver_y, [Ai, Aip], decimal=4)

    @given(floats(max_value=0.), floats(0.01, 9.0))
    def testAiryAnalyticalNegative(self, x0, intervalSize):
        """Tests Airy ODE y'' = xy against analytical solution with a negative
        step"""

        # Assumptions: the initial condition time is not too big
        assume(abs(x0) < 1e4)

        # Initial conditions taken from the airy function at the initial time
        Ai, Aip, Bi, Bip = airy(x0)
        y0 = [Ai, Aip]

        # Own solver build (force double precision)
        RK4_y0 = np.array([[y0]]).astype(np.float64)
        RK4_solver = RK4Solver(x0, RK4_y0, -self.dx, self.RK4_functions)

        # Evolution of the system
        RK4_solver_y = RK4_solver.solve(RK4_solver.x0 - intervalSize)[0, 0, :]
        Ai, Aip, Bi, Bip = airy(RK4_solver.x0)

        # Test that both data are almost the same, up to 4 decimal places
        np.testing.assert_almost_equal(RK4_solver_y, [Ai, Aip], decimal=4)


if __name__ == '__main__':
    # Run all the tests
    unittest.main()
