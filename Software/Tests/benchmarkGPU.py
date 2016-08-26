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


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Black hole constants
    spin = 0.00000001
    innerDiskRadius = 9
    outerDiskRadius = 20

    # Create the black hole
    blackHole = BlackHole(spin, innerDiskRadius, outerDiskRadius)

    # Camera position
    camR = 30
    camTheta = 1.511
    camPhi = 0

    # Camera lens properties
    camFocalLength = 3
    camSensorSize = (2, 2)       # (Height, Width)

    if len(sys.argv) < 5:
        print("Usage: python benchmarkGPU.py outputPath minSide maxSide step")
        sys.exit()

    # Benchmark parameters
    outputPath = sys.argv[1]
    start = int(sys.argv[2])
    stop = int(sys.argv[3])
    step = int(sys.argv[4])

    output = open(outputPath, 'w')

    # Print CSV header
    print("Number of pixels, GPU time", file=output)

    # Run the benchmark!
    for side in range(start, stop, step):
        # Create the camera with the current side
        camSensorShape = (side, side)  # (Rows, Columns)

        # Create the camera and the metric with the constants above
        camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                        camSensorSize)
        kerr = KerrMetric(camera, blackHole)

        # Set camera's speed (it needs the kerr metric constants)
        camera.setSpeed(kerr, blackHole)

        # Create the raytracer
        rayTracer = RayTracer(camera, kerr, blackHole)

        # Actual computation
        rayTracer.rayTrace(-90, stepsPerKernel=90, resolution=-1)
        rayTracer.synchronise()

        # Print results both to the CSV file and to the standard output
        currentData = "{}, {:.10f}".format(side*side, rayTracer.totalTime)
        print(currentData, file=output)
