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
sys.path.append('../')
from Raytracer import universe, Camera


if __name__ == '__main__':
    # Camera position
    camR = 40
    camTheta = 1.511
    camPhi = 0

    # Camera lens properties
    camFocalLength = 3
    camSensorSize = (2, 2)       # (Height, Width)
    camSensorShape = (1000, 1000)  # (Rows, Columns)

    # Set black hole spin
    universe.spin = .999
    universe.accretionDisk.innerRadius = 9
    universe.accretionDisk.outerRadius = 20

    # Create a camera
    camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                     camSensorSize)

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
        # Change the camera sensor
        camera.sensorShape = (side, side)  # (Rows, Columns)

        _, time = camera.shoot(finalTime=-150)

        # Print results both to the CSV file and to the standard output
        currentData = "{}, {:.10f}".format(side*side, time)
        print(currentData, file=output)
