import unittest
import sys

import numpy as np

from hypothesis import given
from hypothesis.strategies import floats, tuples, sampled_from, integers

from scipy.integrate import ode
from scipy.special import airy

sys.path.append('../RK4')
from rk4 import RK4Solver


class TestRK4(unittest.TestCase):
    """Test suite for RungeKutta4 solver"""
    def setUp(self):
        # Initial conditions for the airy function defined in SciPy
        self.x0 = -100
        self.y0 = [0.040241238486441955, 0.99626504413279049]
        self.dx = 0.001

        # Definition of system for own solver
        self.RK4_functions = ["y[1]", "x*y[0]"]


    # @given(floats(-10., 0.), tuples(floats(-10., 10.), floats(-10., 10.)))
    # def testAiryNumerical(self, x0, y0):
    #     "Tests Airy ODE y'' = xy against SciPy solver"
    #
    #     # Definition of system for SciPy solver
    #     def SCI_functions(x, y):
    #         return [y[1], x*y[0]]
    #
    #     # Own solver build
    #     RK4_y0 = np.array([[y0]]).astype(np.float64)
    #     RK4_solver = RK4Solver(x0, RK4_y0, self.dx, self.RK4_functions)
    #
    #     # SciPy solver build
    #     SCI_solver = ode(SCI_functions).set_integrator('dopri5',
    #                                                    first_step=self.dx)
    #
    #     SCI_solver.set_initial_value(y0, x0)
    #
    #     # Evolution of the system with both solvers
    #     while RK4_solver.x0 < x0+2:
    #         RK4_solver_y = RK4_solver.solve()[0, 0, :]
    #         SCI_solver.integrate(RK4_solver.x0)
    #
    #         # Debugggggging
    #         print(RK4_solver_y)
    #         print(SCI_solver.y, "\n")
    #
    #         # # Test that both data are almost the same, up to 3 decimal places
    #         # np.testing.assert_almost_equal(RK4_solver_y, SCI_solver.y,
    #         #                                decimal=3)


    # # @given(floats(0.01, 2.0), integers(1, 1000))
    # def testAiryAnalytical(self):
    #     "Tests Airy ODE y'' = xy against analytical solution"
    #
    #     # Own solver build (force double precision)
    #     RK4_y0 = np.array([[self.y0]]).astype(np.float64)
    #     RK4_solver = RK4Solver(self.x0, RK4_y0, self.dx, self.RK4_functions)
    #
    #     # Evolution of the system
    #     while RK4_solver.x0 < 0:
    #         RK4_solver_y = RK4_solver.solve()[0, 0, :]
    #         Ai, Aip, Bi, Bip = airy(RK4_solver.x0)
    #         print(RK4_solver.step)
    #
    #         # Debugggggging
    #         print(RK4_solver_y)
    #         print([Ai, Aip], "\n\n")
    #
    #         # Test that both data are almost the same, up to 3 decimal places
    #         # np.testing.assert_almost_equal(RK4_solver_y, [Ai, Aip], decimal=3)

    def testSciPyAnalytical(self):
        "Tests Airy ODE y'' = xy against analytical solution"

        # Definition of system for SciPy solver
        def SCI_functions(x, y):
            return [y[1], x*y[0]]

        # SciPy solver build
        SCI_solver = ode(SCI_functions).set_integrator('dopri5',
                                                       first_step=self.dx)

        SCI_solver.set_initial_value(self.y0, self.x0)

        # Evolution of the system
        while SCI_solver.t < 0:
            SCI_solver.integrate(SCI_solver.t+self.dx)
            Ai, Aip, Bi, Bip = airy(SCI_solver.t)

            # Debugggggging
            print(SCI_solver.y)
            print([Ai, Aip], "\n\n")

            # Test that both data are almost the same, up to 3 decimal places
            # np.testing.assert_almost_equal(RK4_solver_y, [Ai, Aip], decimal=3)


if __name__ == '__main__':
    # Run all the tests
    unittest.main()
