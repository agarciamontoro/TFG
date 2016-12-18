import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

"""
Put
    int p1 =floor(fmod(theta+Pi, Pi) * 8.0 / (Pi));
    int p2 = floor(fmod(phi+2*Pi, 2*Pi) * 10.0 / (2*Pi));

    image[0] = image[1] = image[2] = 0;

    if((p1 ^ p2) & 1){
        image[0] = phi/(2*Pi);
        image[1] = 0.5;
        image[2] = 1-phi/(2*Pi);
    }
in the case HORIZON of image_transformation.cu
"""

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
universe.accretionDisk.innerRadius = 2
universe.accretionDisk.outerRadius = 1

# Create a camera
camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

sphere = "../../Data/Textures/milkyWay.png"
disk = "../../Data/Textures/adisk.png"

suffix = 0
amp = 1

suffix = 0
amp = 1

for angle in np.arange(0, 2*np.pi, 0.1):
    suffix += 1

    camera.phi = angle
    camera.speed = 0

    texturedImage, _ = camera.shoot(diskPath=disk, spherePath=sphere)
    texturedImage.save("../../Documentation/Presentation/gfx/patchedShadow" + "%d" % suffix + ".png")
