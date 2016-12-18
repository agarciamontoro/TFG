import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

# Camera position
camR = 40
camTheta = 1.7
camPhi = 0

# Camera lens properties
camFocalLength = 5
camSensorShape = (900, 1600)  # (Rows, Columns)
camSensorSize = (9, 16)   # (Height, Width)

# Set black hole spin
universe.spin = .999

# Create a camera
camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

sphere = "../../Data/Textures/milkyWay.png"
disk = "../../Data/Textures/adisk.png"

suffix = 0
amp = 1

for angle in np.arange(0, np.pi, 0.05):
    suffix += 1

    camera.phi = angle
    camera.yaw = -angle / 2
    camera.r = 45 - 38*(angle / (np.pi))

    print(camera.phi, camera.theta)

    texturedImage, _ = camera.shoot(diskPath=disk, spherePath=sphere)
    texturedImage.save("../../Documentation/Presentation/gfx/cinema02_" + "%d" % suffix + ".png")
