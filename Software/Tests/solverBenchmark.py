import sys
import numpy as np

sys.path.append('../RK4')
from rk4 import RK4Solver

if __name__ == '__main__':
    RK4_functions = ["y[1]", "x*y[0]"]

    x0 = -10.
    xend = -5.

    for blockSize in range(1, 501):
        y0 = np.empty((blockSize, blockSize, 2)).astype(np.float64)
        y0[:, :, :] = 1.

        # Own solver build
        RK4_solver = RK4Solver(x0, y0, 0.001, RK4_functions)

        # Evolution of the system with both solvers
        RK4_solver_y = RK4_solver.solve(xend)

        print(RK4_solver.totalTime)
