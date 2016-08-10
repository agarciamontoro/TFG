import unittest
import sys

import numpy as np

from hypothesis import given
from hypothesis.strategies import floats, tuples, sampled_from

from scipy.integrate import ode

sys.path.append('../RK4')
from rk4 import RK4Solver


class TestRK4(unittest.TestCase):
    """Test suite for RungeKutta4 solver"""

    @given(floats(), tuples(floats(), floats()),
           sampled_from(np.arange(0.02, 1, 0.01)), floats(0, 1))
    def testSystem2(self, x0, y0, dx, constant):
        RK4_functions = ["y[1]",
                         str(constant)+"*y[0]"]

        def SCI_functions(x, y, arg1):
            return [y[1], arg1*y[0]]

        RK4_y0 = np.array([[list(y0)]])
        RK4_solver = RK4Solver(x0, RK4_y0, dx, RK4_functions)

        SCI_solver = ode(SCI_functions).set_integrator('dopri5')
        SCI_solver.set_initial_value(y0, x0).set_f_params(constant)

        for i in range(100):
            RK4_solver_y = RK4_solver.solve()[0, 0, :]
            SCI_solver.integrate(SCI_solver.t+dx)

            np.testing.assert_almost_equal(RK4_solver_y.astype(np.float32),
                                           SCI_solver.y.astype(np.float32),
                                           decimal=1)

if __name__ == '__main__':
    unittest.main()
