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
    spin = 0.00000001
    innerDiskRadius = 9
    outerDiskRadius = 20

    # Camera position
    camR = 30
    camTheta = 1.511
    camPhi = 0

    # Camera lens properties
    camFocalLength = 3
    camSensorShape = (400, 400)  # (Rows, Columns)
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
    print(rayTracer.totalTime)

    # rayTracer.generate3Dscene(-1, 10)
    # rayTracer.plotScene()

    rayTracer.synchronise()
    rayTracer.plotImage()

    print("End")
