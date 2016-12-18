import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

# Camera position
camR = 40
camTheta = 0.705493850862
camPhi = 0

# Camera lens properties
camFocalLength = 10
camSensorShape = (1, 1)  # (Rows, Columns)
camSensorSize = (1, 1)   # (Height, Width)

# Set black hole spin
universe.spin = .999

# Create a camera
camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

camera.slicedShoot().plot()
