import numpy as np
import sys
sys.path.append('../')

from Raytracer import universe, Camera

# Camera position
camR = 40
camTheta = 1.511
camPhi = 0

# Camera lens properties
camFocalLength = 20
camSensorShape = (10, 10)  # (Rows, Columns)
camSensorSize = (6, 6)   # (Height, Width)

# Set black hole spin
universe.spin = .999
universe.accretionDisk.innerRadius = 7
universe.accretionDisk.outerRadius = 20
# Create a camera
camera1 = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                 camSensorSize)

# Create another camera
camera2 = Camera(camR, camTheta, camPhi, camFocalLength, (4000, 4000),
                 camSensorSize)

# # Make an sliced shoot; i.e., store all the intermediate steps in order to
# # plot a 3D scene
# plot3D = camera1.slicedShoot(finalTime=-60, slicesNum=10000)
# plot3D.plot()
#
# disk = '../../Data/Textures/patchdisk.png'
# sphere = '../../Data/Textures/milkyWay.png'
# camera1.sensorShape = (3000, 3000)
# texturedImage, _ = camera1.shoot(diskPath=disk, spherePath=sphere)
# texturedImage.plot()

# # Plot only one geodesic, indexing it with the pixel row,col
# plot3D.geodesic(2, 4).plot()
# # You can even plot a snapshot, which may be not that interesting, though...
# plot3D.snapshot(1).plot()
# #
# Make a proper photography!
camera2.yaw = -0.06
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
