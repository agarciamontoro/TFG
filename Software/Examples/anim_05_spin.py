import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

"""
Put
    p1 = rNormalized;
    p2 = floor(fmod(phi+2*Pi, 2*Pi) * 26.0 / (2*Pi));

    image[0] = image[1] = image[2] = 1;

    if((p1 ^ p2) & 1){
        image[0] = 1;
        image[1] = 0;
        image[2] = 0;
    }
in the DISK case in image_transformation.cu
"""

# Camera position
camR = 10
camTheta = 0.001
camPhi = 0

# Camera lens properties
camFocalLength = 8
camSensorShape = (900, 1600)  # (Rows, Columns)
camSensorSize = (9, 16)   # (Height, Width)

# Set black hole spin
universe.spin = .005
universe.accretionDisk.innerRadius = 1
universe.accretionDisk.outerRadius = 50

# Create a camera
camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

sphere = "../../Data/Textures/milkyWay.png"
disk = "../../Data/Textures/adisk.png"

suffix = 0
amp = 1

for spin in np.arange(0.000001, 0.999999, 0.02):
    suffix += 1

    universe.spin = spin

    print(universe.spin)

    texturedImage, _ = camera.shoot(diskPath=disk, spherePath=sphere)
    texturedImage.save("../../Documentation/Presentation/gfx/spin" + "%d" % suffix + ".png")
