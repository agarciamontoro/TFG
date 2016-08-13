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

        # Test that both data are almost the same, up to 3 decimal places
        np.testing.assert_almost_equal(RK4_solver_y, SCI_solver.y, decimal=5)


    @given(floats(0.01, 9.0))
    def testAiryAnalytical(self, intervalSize):
        "Tests Airy ODE y'' = xy against analytical solution"

        # Initial conditions for the airy function defined in SciPy
        x0 = -10
        y0 = [0.040241238486441955, 0.99626504413279049]

        # Own solver build (force double precision)
        RK4_y0 = np.array([[y0]]).astype(np.float64)
        RK4_solver = RK4Solver(x0, RK4_y0, self.dx, self.RK4_functions)

        # Evolution of the system
        RK4_solver_y = RK4_solver.solve(RK4_solver.x0 + intervalSize)[0, 0, :]
        Ai, Aip, Bi, Bip = airy(RK4_solver.x0)

        # Test that both data are almost the same, up to 3 decimal places
        np.testing.assert_almost_equal(RK4_solver_y, [Ai, Aip], decimal=6)

if __name__ == '__main__':
    # Run all the tests
    unittest.main()
