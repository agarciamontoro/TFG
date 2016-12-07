from ..universe import universe

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d


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


def drawErgoSphere(ax):
    a2 = universe.spinSquared

    # Draw black hole
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    r = (2 + np.sqrt(4 - 4*a2*np.square(np.cos(v)))) / 2

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z)


def drawCameras(ax):
    for camera in universe.cameras:
        d = camera.r + camera.focalLength
        H = camera.sensorSize[0] / 2
        W = camera.sensorSize[1] / 2

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
    ax.plot((0, 0), (0, 0), (-d, d), 'grey')


def drawBlackHole(ax):
    # Draw horizon
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    r = universe.horizonRadius

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='black',
                    edgecolors='white', linewidth=0.15)

    # Draw accretion disk
    circle1 = Circle((0, 0), universe.accretionDisk.innerRadius,
                     facecolor='none')
    circle2 = Circle((0, 0), universe.accretionDisk.outerRadius,
                     facecolor='none')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    art3d.pathpatch_2d_to_3d(circle1, z=0, zdir='z')
    art3d.pathpatch_2d_to_3d(circle2, z=0, zdir='z')


def drawGeodesic(ax, coordinates, colour):
    # Compute cartesian coordinates of the ray
    x, y, z = spher2cart(coordinates)

    # Plot the ray!
    ax.plot(x, y, z, label='Ray', color=colour, linewidth=1.5)


def drawScene(ax):
    drawAxes(ax)
    drawBlackHole(ax)
    drawCameras(ax)
