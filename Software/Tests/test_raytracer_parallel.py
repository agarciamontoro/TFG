import unittest
import sys

sys.path.append('../Raytracer')
from raytracer import RayTracer, Camera
from kerr import BlackHole, KerrMetric
from utilities import override_initial_conditions

class Test_Solver(unittest.TestCase):
    """Test suite for raytracer RK45 solver"""
    def setUp(self):

     # Black hole constants
     spin = 0.9999999999
     innerDiskRadius = 9
     outerDiskRadius = 20

     # Camera position
     camR = 30
     camTheta = 1.511
     camPhi = 0

     # Camera lens properties
     camFocalLength = 3
     camSensorShape = (1000, 1000)  # (Rows, Columns)
     camSensorSize = (2, 2)       # (Height, Width)

     # Create the black hole, the camera and the metric with the constants
     # above
     blackHole = BlackHole(spin, innerDiskRadius, outerDiskRadius)
     camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                     camSensorSize)
     kerr = KerrMetric(camera, blackHole)

     # Set camera's speed (it needs the kerr metric constants)
     camera.setSpeed(kerr, blackHole)

     # Monky-patch the raytracer with the overrider

     RayTracer.override_initial_conditions = override_initial_conditions

     # Create the raytracer!
     self.rayTracer = RayTracer(camera, kerr, blackHole)

    def test_ray(self):

        self.rayTracer.override_initial_conditions(10,1.415,0,1.445,-0.659734)

        return True

if __name__ == '__main__':
    # Run all the tests
    unittest.main()
