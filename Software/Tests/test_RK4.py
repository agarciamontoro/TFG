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

    @given(floats(max_value=0.), tuples(floats(), floats()),
           floats(min_value=0.01), integers(1, 1000))
    def testAiryNumerical(self, x0, y0, dx, steps):
        "Tests Airy ODE y'' = xy against SciPy solver"

        # Definition of system for own solver
        RK4_functions = ["y[1]",
                         "x*y[0]"]

        # Definition of system for SciPy solver
        def SCI_functions(x, y):
            return [y[1], x*y[0]]

        # Own solver build
        RK4_y0 = np.array([[y0]]).astype(np.float32)
        RK4_solver = RK4Solver(x0, RK4_y0, dx, RK4_functions)

        # SciPy solver build
        SCI_solver = ode(SCI_functions).set_integrator('dopri5')
        SCI_solver.set_initial_value(y0, x0)

        # Evolution of the system with both solvers
        for i in range(steps):
            RK4_solver_y = RK4_solver.solve()[0, 0, :]
            SCI_solver.integrate(SCI_solver.t+dx)

            # Test that both data are almost the same, up to 5 decimal places
            np.testing.assert_almost_equal(RK4_solver_y.astype(np.float32),
                                           SCI_solver.y.astype(np.float32),
                                           decimal=5)

    @given(floats(min_value=0.01), integers(1, 1000))
    def testAiryAnalytical(self, dx, steps):
        "Tests Airy ODE y'' = xy against analytical solution"
        # Definition of system for own solver
        RK4_functions = ["y[1]",
                         "x*y[0]"]

        # Initial conditions for the airy function defined in SciPy
        x0 = -10
        y0 = (0.04024123848644319,
              0.99626504413279049)

        # Own solver build
        RK4_y0 = np.array([[y0]]).astype(np.float32)
        RK4_solver = RK4Solver(x0, RK4_y0, dx, RK4_functions)

        # Evolution of the system
        for i in range(steps):
            Ai, Aip, Bi, Bip = airy(RK4_solver.x0+RK4_solver.step)
            RK4_solver_y = RK4_solver.solve()[0, 0, :]
            
            # Test that both data are almost the same, up to 3 decimal places
            np.testing.assert_almost_equal(RK4_solver_y.astype(np.float32),
                                           np.float32([Ai, Aip]),
                                           decimal=3)



if __name__ == '__main__':
    # # Run all the tests
    unittest.main()
