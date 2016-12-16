import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

# Camera position
camR = 12
camTheta = 1.511
camPhi = 0

# Camera lens properties
camFocalLength = 10
camSensorShape = (900, 1600)  # (Rows, Columns)
camSensorSize = (9, 16)   # (Height, Width)

# Set black hole spin
universe.spin = .0000000000000001
universe.accretionDisk.innerRadius = 3
universe.accretionDisk.outerRadius = 2

# Create a camera
camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

sphere = "../../Data/Textures/milkyWay.png"
disk = "../../Data/Textures/adisk.png"

suffix = 0
amp = 1

texturedImage, _ = camera.shoot(diskPath=disk, spherePath=sphere)
texturedImage.plot()
texturedImage.save("asdad.png")
