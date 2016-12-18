import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

# Camera position
camR = 20
camTheta = 1.511
camPhi = 0

# Camera lens properties
camFocalLength = 5
camSensorShape = (900, 1600)  # (Rows, Columns)
camSensorSize = (9, 16)   # (Height, Width)

# Set black hole spin
universe.spin = .0000001
universe.accretionDisk.innerRadius = 8
universe.accretionDisk.outerRadius = 7

# Create a camera
camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

sphere = "../../Data/Textures/milkyWay.png"
disk = "../../Data/Textures/adisk.png"

suffix = 0
amp = 1

camera.speed = 0

texturedImage, _ = camera.shoot()
texturedImage.plot()
texturedImage.save("complete00.png")
