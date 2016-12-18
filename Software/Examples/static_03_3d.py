import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

# Camera position
camR = 10
camTheta = np.pi/2
camPhi = 0

# Camera lens properties
camFocalLength = 1
camSensorShape = (4, 10)  # (Rows, Columns)
camSensorSize = (1, 1)   # (Height, Width)

# Set black hole spin
universe.spin = .9999
universe.accretionDisk.innerRadius = 8
universe.accretionDisk.outerRadius = 18

# Create a camera
camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

sphere = "../../Data/Textures/milkyWay.png"
disk = "../../Data/Textures/adisk.png"

suffix = 0
amp = 1

camera.speed = 0

texturedImage = camera.slicedShoot(slicesNum=50000)
texturedImage.plot()
