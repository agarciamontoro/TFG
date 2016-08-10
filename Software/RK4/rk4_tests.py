from rk4 import RK4Solver
import numpy as np

if __name__ == "__main__":
    x0 = -1
    y0 = np.array([[[1, 1]], [[2, 2]]])
    dx = 0.02

    functions = ["y[1]",
                 "-25*y[0]"]

    solver = RK4Solver(x0, y0, dx, functions)

    for i in range(100):
        print(solver.solve())
