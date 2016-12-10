import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

# Camera position
camR = 35
camTheta = 0
camPhi = 0

# Camera lens properties
camFocalLength = 4
camSensorShape = (5, 5)  # (Rows, Columns)
camSensorSize = (6, 9)       # (Height, Width)

# Set black hole spin
universe.spin = .999

# Create a camera
camera1 = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                 camSensorSize)

# Create another camera
camera2 = Camera(camR, camTheta, camPhi, camFocalLength, (1000, 1500),
                 camSensorSize)

# # Make an sliced shoot; i.e., store all the intermediate steps in order to
# # plot a 3D scene
# plot3D = camera1.slicedShoot(slicesNum=5000)
# plot3D.plot()
#
# # Plot only one geodesic, indexing it with the pixel row,col
# plot3D.geodesic(5, 5).plot()
# # You can even plot a snapshot, which may be not that interesting, though...
# plot3D.snapshot(1).plot()
# #
# # Make a proper photography!
# photo = camera2.shoot()
# photo.plot()

# Load the textures
disk = '../../Data/Textures/patchdisk.png'
sphere = '../../Data/Textures/milkyWay.png'
texturedImage = camera2.shoot(diskPath=disk, spherePath=sphere)
texturedImage.plot()

# for theta in np.arange(0.8, np.pi - 0.8, 0.005):
#     camera2.theta = theta
#     texturedImage = camera2.shoot(diskPath=disk, spherePath=sphere)
#     texturedImage.save("pruebita" + "%.3f" % theta + ".png")
