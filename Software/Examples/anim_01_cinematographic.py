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

for angle in np.arange(0, 2 * np.pi, 0.05):
    suffix += 1

    camera.phi = angle
    camera.theta = (np.sin(angle)/2 + 0.5) * (np.pi - 2 * amp) + amp

    print(camera.phi, camera.theta)

    texturedImage, _ = camera.shoot(diskPath=disk, spherePath=sphere)
    texturedImage.save("../../Documentation/Presentation/gfx/cinema01_" + "%d" % suffix + ".png")
