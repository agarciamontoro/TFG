import os
import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def drawRay(ax, filePath):
    # Retrieve ray points
    sphericalPoints = genfromtxt(filePath, delimiter=',')

    # Retrieve the actual data
    r = sphericalPoints[:, 3]
    theta = sphericalPoints[:, 4]
    phi = sphericalPoints[:, 5]

    cosT = np.cos(theta)
    sinT = np.sin(theta)
    cosP = np.cos(phi)
    sinP = np.sin(phi)

    x = r * sinT * cosP
    y = r * sinT * sinP
    z = r * cosT

    ax.plot(x, y, z, label='Ray0')


def drawRays(ax, filePath):
    # Retrieve ray points
    data = genfromtxt(filePath, delimiter=',')

    for i in range(0, 100, 10):
        ray = data[data[:, 0] == i, :]

        ray = ray[ray[:, 2].argsort()[::-1]]

        print(ray)

        r = ray[:, 3]
        theta = ray[:, 4]
        phi = ray[:, 5]

        cosT = np.cos(theta)
        sinT = np.sin(theta)
        cosP = np.cos(phi)
        sinP = np.sin(phi)

        x = r * cosT * sinP
        y = r * sinT * sinP
        z = r * cosP

        ax.plot(x, y, z, label='Ray0', c='blue')


def drawCamera(ax):
    camR = 100
    camTheta = np.pi/2
    camPhi = 0


    camX = camR * np.sin(camTheta) * np.cos(camPhi)
    camY = camR * np.sin(camTheta) * np.sin(camPhi)
    camZ = camR * np.cos(camTheta)

    ax.scatter(camX, camY, camZ, s=100, c='red')

    x = [1, 1, -1, -1]
    y = [1, -1, -1, 1]
    z = [-1, -1, -1, -1]
    verts = [(x[i], y[i], z[i]) for i in range(4)]
    # ax.add_collection3d(Poly3DCollection(verts))


def drawAxes(ax, d=150):
    ax.plot((-d, d), (0, 0), (0, 0), 'grey')
    ax.plot((0, 0), (-d, d), (0, 0), 'grey')
    ax.plot((0, 0), (0, 0), (-d, d), 'gray')


def drawBlackHole(ax, r=5):
    # Draw black hole
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='black')


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

if __name__ == '__main__':
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.set_axis_off()

    ax.set_xlim3d(-25, 25)
    ax.set_ylim3d(-25, 25)
    ax.set_zlim3d(-25, 25)

    # axisEqual3D(ax)

    drawAxes(ax)
    drawBlackHole(ax)
    drawCamera(ax)

    # drawRay(ax, "Data/rayPositions.csv")
    # drawRay(ax, "Data/middleRay.csv")
    # drawRays(ax, "Data/rays.csv")

    # for fileName in absoluteFilePaths("Data/Spin00001"):
    #     if fileName.endswith(".csv"):
    #         drawRay(ax, fileName)
    #
    drawRay(ax, "Data/Spin00001/ray00.csv")
    drawRay(ax, "Data/Spin00001/ray10.csv")
    drawRay(ax, "Data/Spin00001/ray20.csv")
    # drawRay(ax, "Data/Spin00001/ray30.csv")
    drawRay(ax, "Data/Spin00001/ray40.csv")
    drawRay(ax, "Data/Spin00001/ray50.csv")
    drawRay(ax, "Data/Spin00001/ray60.csv")
    # drawRay(ax, "Data/Spin00001/ray70.csv")
    drawRay(ax, "Data/Spin00001/ray80.csv")
    drawRay(ax, "Data/Spin00001/ray90.csv")
    drawRay(ax, "Data/Spin00001/ray99.csv")

    # ax.legend()

    plt.show()
