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

SPHERE = 0
DISK = 1
HORIZON = 2

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Black hole constants
    spin = 0.8
    innerDiskRadius = 1
    outerDiskRadius = 1

    # Camera position
    camR = 10
    camTheta = 1.415
    camPhi = 0

    # Camera lens properties
    camFocalLength = 3
    camSensorShape = (100, 100)  # (Rows, Columns)
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
    rayTracer.override_initial_conditions(10,1.415,0,1.445,-0.659734)
    # Draw the image
    #rayTracer.rayTrace(-90, kernelCalls=1)
    #print(rayTracer.totalTime)
    #rayTracer.synchronise()
    # # np.savetxt("data.csv", rayTracer.systemState[20, 20, :])
    #rayTracer.plotImage()

    # # Generate the 3D scene
    rayTracer.generate3Dscene(-10, 300)
    np.savetxt("./cosa.csv", np.transpose(rayTracer.plotData[50, 50, :,:]),fmt="%.15f", delimiter=",",newline='\n')
    #rayTracer.plotScene()
