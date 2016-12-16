from .Utils.draw import drawScene, drawGeodesic

import numpy as np
from matplotlib import pyplot as plt

SPHERE = 0
DISK = 1
HORIZON = 2


class Geodesic:
    def __init__(self, status, coordinates, colour='royalBlue'):
        self.status = SPHERE
        self.coordinates = coordinates

        # Detect if the ray collided with the disk, remove the following steps
        # and change its colour
        indicesDisk = np.where(status == DISK)[0]

        if indicesDisk.size > 0:
            self.status = DISK

            firstCollision = indicesDisk[0]
            self.coordinates = coordinates[:firstCollision, :]

        # Detect if the ray entered the horizon, remove the following steps
        # and change its colour
        indicesCollision = np.where(status == HORIZON)[0]

        if indicesCollision.size > 0:
            self.status = HORIZON

            firstCollision = indicesCollision[0]
            self.coordinates = coordinates[:firstCollision, :]

        # Set colour
        if self.status == SPHERE:
            self.colour = 'royalBlue'
        elif self.status == HORIZON:
            self.colour = 'maroon'
        else:
            self.colour = 'darkolivegreen'

    def plot(self, ax=None):
        showPlot = False

        if not ax:
            showPlot = True

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
            drawScene(ax)

        drawGeodesic(ax, self.coordinates, self.colour)

        if showPlot:
            # Show the plot
            plt.show()


class CongruenceSnapshot:
    def __init__(self, status, coordinates, texels=None):
        self.status = status
        self.coordinates = coordinates
        self.texels = texels

        self.congruenceMatrixRows = self.status.shape[0]
        self.congruenceMatrixCols = self.status.shape[1]
        self.numPixels = self.congruenceMatrixRows * self.congruenceMatrixCols

        self.colors = [
            [1, 1, 1],  # Sphere
            [1, 0, 0],  # Disk
            [0, 0, 0]   # Horizon
        ]

    def plot(self):
        plt.figure()
        plt.axis('off')

        if self.texels is None:
            image = np.empty((self.congruenceMatrixRows,
                              self.congruenceMatrixCols,
                              3))

            for row in range(self.congruenceMatrixRows):
                for col in range(self.congruenceMatrixCols):
                    status = self.status[row, col]

                    image[row, col, :] = self.colors[status]

            plt.imshow(image)
        else:
            plt.imshow(self.texels)

        plt.show()

    def save(self, path):
        fig = plt.figure()
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        plt.axis('off')

        if self.texels is None:
            image = np.empty((self.congruenceMatrixRows,
                              self.congruenceMatrixCols,
                              3))

            for row in range(self.congruenceMatrixRows):
                for col in range(self.congruenceMatrixCols):
                    status = self.status[row, col]

                    image[row, col, :] = self.colors[status]

            plt.imshow(image)
        else:
            plt.imshow(self.texels)

        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)


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

    def geodesic(self, row, col):
        return Geodesic(self.status[row, col, :],
                        np.transpose(self.coordinates[row, col, :, :]))

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
        drawScene(ax)

        # Draw the rays
        for row in range(0, self.congruenceMatrixRows):
            for col in range(0, self.congruenceMatrixCols):
                self.geodesic(row, col).plot(ax)

        # Add a legend
        # ax.legend()

        # Show the plot
        plt.show()
