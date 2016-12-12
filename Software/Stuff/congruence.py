import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

# Camera position
camR = 6
camTheta = 0.00001
camPhi = 0

# Camera lens properties
camFocalLength = 5
camSensorShape = (5, 5)  # (Rows, Columns)
camSensorSize = (6, 9)       # (Height, Width)

# Set black hole spin
universe.spin = .999
universe.accretionDisk.innerRadius = 1
# Create a camera
camera1 = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                 camSensorSize)

# Create another camera
camera2 = Camera(camR, camTheta, camPhi, camFocalLength, (1600, 2400),
                 camSensorSize)

# # Make an sliced shoot; i.e., store all the intermediate steps in order to
# # plot a 3D scene
# camera1.pitch = 0.2
# plot3D = camera1.slicedShoot(slicesNum=1000)
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


# # Load the textures
disk = '../../Data/Textures/griddisk.png'
sphere = '../../Data/Textures/milkyWay.png'
texturedImage, _ = camera2.shoot(diskPath=disk, spherePath=sphere)
texturedImage.plot()


# suffix = 0
#
# for pitch in np.arange(-np.pi/2, np.pi/2, 0.05):
#     suffix += 1
#     camera2.pitch = pitch
#     texturedImage = camera2.shoot(diskPath=disk, spherePath=sphere)
#     texturedImage.save("pruebita" + "%03d" % suffix + ".png")
