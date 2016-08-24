import os
import sys
import numpy as np
from numpy import sin, cos, arccos, arctan, arctan2, sqrt
from numpy import pi as Pi

from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

# Import the raytracer
sys.path.append('../Raytracer')
from raytracer import RayTracer, Camera
from kerr import BlackHole, KerrMetric


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def plotImage(status):
    # Start figure
    fig = plt.figure()

    image = np.empty((301, 301, 3))

    for row in range(0, 301):
        for col in range(0, 301):
            image[row, col, :] = drawRayImg(status[row, col])

    plt.imshow(image)
    plt.show()


def plotScene(plotData, status, camera, blackHole):
    # Start figure
    fig = plt.figure()

    # Start 3D plot
    ax = fig.gca(projection='3d')
    ax.set_axis_off()

    # Set axes limits
    ax.set_xlim3d(-25, 25)
    ax.set_ylim3d(-25, 25)
    ax.set_zlim3d(-25, 25)

    # Draw the scene
    drawAxes(ax)
    drawBlackHole(ax, blackHole)
    drawErgoSphere(ax, blackHole)
    drawCamera(ax, camera)

    for row in range(0, 301, 20):
        for col in range(0, 301, 20):
            ray = np.transpose(plotData[row, col, :, :])
            drawRay(ax, ray, status[row, col, :])

    ray = np.transpose(plotData[90, 70, :, :])
    drawRay(ax, ray, status[90, 70, :])

    # Add a legend
    # ax.legend()

    # Show the plot
    plt.show()

SPHERE = 0
DISK = 1
HORIZON = 2


def drawRayImg(status):
    if status == DISK:
        return [1, 0, 0]

    if status == HORIZON:
        return [0, 0, 0]

    if status == SPHERE:
        return [1, 1, 1]



def spher2cart(points):
    # Retrieve the actual data
    r = points[:, 0]
    theta = points[:, 1]
    phi = points[:, 2]

    cosT = np.cos(theta)
    sinT = np.sin(theta)
    cosP = np.cos(phi)
    sinP = np.sin(phi)

    x = r * sinT * cosP
    y = r * sinT * sinP
    z = r * cosT

    return x, y, z


def drawErgoSphere(ax, blackHole):
    a2 = blackHole.a2

    # Draw black hole
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    r = (2 + np.sqrt(4 - 4*a2*np.square(np.cos(v)))) / 2

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z)


def drawRay(ax, ray, status):
    rayColor = 'black'

    badRows = np.where(status == 0)[0]
    if badRows.size > 0:
        firstBadRow = badRows[0]
        ray = ray[:firstBadRow, :]
        rayColor = 'red'

    x, y, z = spher2cart(ray)

    ax.plot(x, y, z, label='Ray', color=rayColor)


def drawCamera(ax, cam):
    d = cam.r + cam.focalLength
    H = cam.sensorSize[0] / 2
    W = cam.sensorSize[1] / 2

    points = np.array([
        [d, W, H],
        [d, -W, H],
        [d, -W, -H],
        [d, W, -H],
        [d, W, H]
    ])

    ax.plot(points[:, 0], points[:, 1], points[:, 2])


def drawAxes(ax, d=150):
    ax.plot((-d, d), (0, 0), (0, 0), 'grey')
    ax.plot((0, 0), (-d, d), (0, 0), 'grey')
    ax.plot((0, 0), (0, 0), (-d, d), 'gray')


def drawBlackHole(ax, blackHole):
    r = (2 + np.sqrt(4 - 4*blackHole.a2)) / 2

    # Draw black hole
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='black',
                    edgecolors='white')


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Black hole constants
    spin = 0.999
    innerDiskRadius = 9
    outerDiskRadius = 20

    # Camera position
    camR = 74
    camTheta = 1.511
    camPhi = 0

    # Camera lens properties
    camFocalLength = 3
    camSensorShape = (301, 301)  # (Rows, Columns)
    camSensorSize = (2, 2)       # (Height, Width)

    # Create the black hole, the camera and the metric with the constants
    # above
    blackHole = BlackHole(spin, innerDiskRadius, outerDiskRadius)
    camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                    camSensorSize)
    kerr = KerrMetric(camera, blackHole)

    # Set camera's speed (it needs the kerr metric constants)
    camera.setSpeed(kerr, blackHole)

    # Create the raytracer!
    rayTracer = RayTracer(camera, kerr, blackHole)
    rayTracer.rayTrace(-90, stepsPerKernel=1)

    status = rayTracer.getStatus()
    plotImage(status)

    print("End")
