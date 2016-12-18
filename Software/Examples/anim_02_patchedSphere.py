import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

"""
Put
    int p1 =floor(fmod(theta+Pi, Pi) * 20.0 / (Pi));
    int p2 = floor(fmod(phi+2*Pi, 2*Pi) * 20.0 / (2*Pi));

    image[1] = image[2] = 0;

    if((p1 ^ p2) & 1)
        image[0] = 1;
    else{
        image[0] = 1;
        image[1] = image[2] = 1;
    }
in the case SPHERE of image_transformation.cu
"""

# Camera position
camR = 40
camTheta = np.pi/2
camPhi = 0

# Camera lens properties
camFocalLength = 5
camSensorShape = (900, 1600)  # (Rows, Columns)
camSensorSize = (9, 16)   # (Height, Width)

# Set black hole spin
universe.spin = .9999

universe.accretionDisk.innerRadius = 2
universe.accretionDisk.outerRadius = 1

# Create a camera
camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

sphere = "../../Data/Textures/milkyWay.png"
disk = "../../Data/Textures/adisk.png"

suffix = 0
amp = 1

for distance in np.arange(40, universe.horizonRadius, -0.5):
    suffix += 1

    camera.r = distance

    texturedImage, _ = camera.shoot(diskPath=disk, spherePath=sphere)
    texturedImage.save("pruebita" + "%03d" % suffix + ".png")
