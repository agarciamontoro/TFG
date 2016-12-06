from Raytracer import universe, Camera

# Camera position
camR = 30
camTheta = 1.511
camPhi = 0

# Camera lens properties
camFocalLength = 3
camSensorShape = (1000, 1000)  # (Rows, Columns)
camSensorSize = (2, 2)       # (Height, Width)

universe.spin = 0.999

camera1 = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                camSensorSize)

image = camera1.slicedShoot(slicesNum=10)
image.snapshot(1).plot()
