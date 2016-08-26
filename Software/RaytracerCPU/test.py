import numpy as np
from numpy import sin, cos, arccos, arctan, arctan2, sqrt
from numpy import pi as Pi
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

SPHERE = 0
DISK = 1
HORIZON = 2

if __name__ == '__main__':
    # Start figure
    fig = plt.figure()

    rayStatus = np.genfromtxt('img.csv', delimiter=',')
    imageRows = rayStatus.shape[0]
    imageCols = rayStatus.shape[1]

    image = np.empty((imageRows, imageCols, 3))

    for row in range(0, imageRows):
        for col in range(0, imageCols):
            status = rayStatus[row, col]

            if status == DISK:
                pixel = [1, 0, 0]

            if status == HORIZON:
                pixel = [0, 0, 0]

            if status == SPHERE:
                pixel = [1, 1, 1]

            image[row, col, :] = pixel

    plt.imshow(image)
    plt.show()
