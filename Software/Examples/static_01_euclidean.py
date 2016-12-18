import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

"""
Put
    f[0] = pR;
    f[1] = pTheta / r2;
    f[2] = b / (r2 * sinT2);
    f[3] =  (pTheta2 + (b2 / sinT2)) /(r2 * r)+ 3*b2/(r2*r2);
    f[4] =   ( b2 ) / ( r2* sinT2) * cosT/sinT;
in computeComponent.

Put
"""

# Camera position
camR = 20
camTheta = 1.511
camPhi = 0

# Camera lens properties
camFocalLength = 8
camSensorShape = (900, 1600)  # (Rows, Columns)
camSensorSize = (9, 16)   # (Height, Width)

# Set black hole spin
universe.spin = .0000000000000001
universe.accretionDisk.innerRadius = 5
universe.accretionDisk.outerRadius = 10

# Create a camera
camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

sphere = "../../Data/Textures/milkyWay.png"
disk = "../../Data/Textures/adisk.png"

suffix = 0
amp = 1

texturedImage, _ = camera.shoot(diskPath=disk, spherePath=sphere)
texturedImage.plot()
texturedImage.save("euclidean02.png")
