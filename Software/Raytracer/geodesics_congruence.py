from .universe import universe

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

SPHERE = 0
DISK = 1
HORIZON = 2

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


class CongruenceSnapshot:
    def __init__(self, status, coordinates):
        self.status = status
        self.coordinates = coordinates

        self.congruenceMatrixRows = self.status.shape[0]
        self.congruenceMatrixCols = self.status.shape[1]
        self.numPixels = self.congruenceMatrixRows * self.congruenceMatrixCols

        self.colors = [
            [1, 1, 1],  # Sphere
            [1, 0, 0],  # Disk
            [0, 0, 0]   # Horizon
        ]

    def plot(self):
        # Start figure
        plt.figure()

        image = np.empty((self.congruenceMatrixRows,
                          self.congruenceMatrixCols,
                          3))

        for row in range(0, self.congruenceMatrixRows):
            for col in range(0, self.congruenceMatrixCols):
                status = self.status[row, col]

                image[row, col, :] = self.colors[status]

        plt.imshow(image)
        plt.show()


class Congruence:
    def __init__(self, status, coordinates):
        self.status = status
        self.coordinates = coordinates
        self.congruenceMatrixRows = status.shape[0]
        self.congruenceMatrixCols = status.shape[1]

        self.numPixels = self.congruenceMatrixRows * self.congruenceMatrixCols
        self.numSlices = status.shape[2]

        self.colors = [
            [1, 1, 1],  # Sphere
            [1, 0, 0],  # Disk
            [0, 0, 0]   # Horizon
        ]

    def snapshot(self, instant):
        return CongruenceSnapshot(self.status[:, :, instant],
                                  self.coordinates[:, :, :, instant])

    def plot(self):
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
        self._drawAxes(ax)
        self._drawBlackHole(ax)
        # self._drawErgoSphere(ax)
        # self._drawCamera(ax)

        # Draw the rays
        for row in range(0, self.congruenceMatrixRows):
            for col in range(0, self.congruenceMatrixCols):
                ray = np.transpose(self.coordinates[row, col, :, :])
                self.plotRay(ax, ray, self.coordinates[row, col, :])

        # Add a legend
        # ax.legend()

        # Show the plot
        plt.show()

    def _drawErgoSphere(self, ax):
        a2 = universe.spinSquared

        # Draw black hole
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        r = (2 + np.sqrt(4 - 4*a2*np.square(np.cos(v)))) / 2

        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_wireframe(x, y, z)

    def plotRay(self, ax, ray, status):
        rayColour = 'royalblue'

        # Detect if the ray collided with the disk, remove the following steps
        # and change its colour
        indicesDisk = np.where(status == DISK)[0]
        if indicesDisk.size > 0:
            firstCollision = indicesDisk[0]
            ray = ray[:firstCollision, :]
            rayColour = 'darkolivegreen'

        # Detect if the ray entered the horizon, remove the following steps
        # and change its colour
        indicesCollision = np.where(status == HORIZON)[0]
        if indicesCollision.size > 0:
            firstCollision = indicesCollision[0]
            ray = ray[:firstCollision, :]
            rayColour = 'maroon'

        # Compute cartesian coordinates of the ray
        x, y, z = spher2cart(ray)

        # Plot the ray!
        ax.plot(x, y, z, label='Ray', color=rayColour, linewidth=1.5)

    def _drawCamera(self, ax):
        d = self.camera.r + self.camera.focalLength
        H = self.camera.sensorSize[0] / 2
        W = self.camera.sensorSize[1] / 2

        points = np.array([
            [d, W, H],
            [d, -W, H],
            [d, -W, -H],
            [d, W, -H],
            [d, W, H]
        ])

        ax.plot(points[:, 0], points[:, 1], points[:, 2])

    def _drawAxes(self, ax, d=150):
        ax.plot((-d, d), (0, 0), (0, 0), 'grey')
        ax.plot((0, 0), (-d, d), (0, 0), 'grey')
        ax.plot((0, 0), (0, 0), (-d, d), 'gray')

    def _drawBlackHole(self, ax):
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
#
#
#
#
# class CongruenceSlice(Congruence):
#     def plot(self):
#         self.plotAtInstant(0)
