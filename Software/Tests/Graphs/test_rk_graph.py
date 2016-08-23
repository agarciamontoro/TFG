import matplotlib.pyplot as plt
import os, sys
import numpy as np

sys.path.append('../../RK4')
from rk4 import RK4Solver


if __name__ == '__main__':
    selfDir = os.path.dirname(os.path.abspath(__file__))
    functions = os.path.abspath(os.path.join(selfDir, "system3.cu"))
    dummyData = np.array([[[1, 2], [1, 2]]])

    # Own solver build (force double precision)
    RK4_y0 = np.array([[[1, 0.5, 1]]]).astype(np.float64)
    RK4_solver = RK4Solver(0, RK4_y0, 0.1, functions, additionalData=dummyData)


    # Evolution of the system

    tInit = 0.
    tEnd = -4
    numSteps = 1000
    stepSize = (tEnd - tInit) / numSteps

    plotData = np.empty((numSteps, 3+1))

    # Simulate!
    t = tInit
    for step in range(numSteps):
        # Advance the step
        t += stepSize

        # Solve the system
        x, y, z = RK4_solver.solve(t)[0, 0, :]
        plotData[step, :] = [RK4_solver.x0, x, y, z]

    t = plotData[:, 0]
    x = plotData[:, 1]
    y = plotData[:, 2]
    z = plotData[:, 3]

    axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    axes.set_ylim([-6, 6])

    # red dashes, blue squares and green triangles
    plt.plot(t, x, 'b-',
             t, y, 'r-',
             t, z, 'g-')
    plt.show()
