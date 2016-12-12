import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

# Camera position
camR = 12
camTheta = np.pi/2
camPhi = 0

# Camera lens properties
camFocalLength = 4
camSensorShape = (5, 5)  # (Rows, Columns)
camSensorSize = (6, 6)       # (Height, Width)

# Set black hole spin
universe.spin = .75
universe.accretionDisk.innerRadius = 20
universe.accretionDisk.outerRadius = 5
# Create a camera
camera1 = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                 camSensorSize)

# Create another camera
camera2 = Camera(camR, camTheta, camPhi, camFocalLength, (2000, 2000),
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
# Make a proper photography!
photo, _ = camera2.shoot()
photo.plot()


# # # Load the textures
# disk = '../../Data/Textures/adisk.png'
# sphere = '../../Data/Textures/milkyWay.png'
# texturedImage, _ = camera2.shoot(diskPath=disk, spherePath=sphere)
# texturedImage.plot()


# suffix = 0
#
# for pitch in np.arange(-np.pi/2, np.pi/2, 0.05):
#     suffix += 1
#     camera2.pitch = pitch
#     texturedImage = camera2.shoot(diskPath=disk, spherePath=sphere)
#     texturedImage.save("pruebita" + "%03d" % suffix + ".png")
