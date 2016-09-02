import unittest
import sys
import numpy as np
import numpy.testing as npt
sys.path.append('../Raytracer')
from raytracer import RayTracer, Camera
from kerr import BlackHole, KerrMetric
from utilities import override_initial_conditions, collect_rays

class Test_Solver(unittest.TestCase):
    """Test suite for raytracer RK45 solver"""
    def setUp(self):

     # Black hole constants
     spin = 0.8
     innerDiskRadius = 1
     outerDiskRadius = 1

     # Camera position
     camR = 10
     camTheta = 1.415
     camPhi = 0

     # Camera lens properties
     camFocalLength = 3
     camSensorShape = (10, 10)  # (Rows, Columns)
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
     RayTracer.collect_rays = collect_rays

     # Create the raytracer!
     self.rayTracer = RayTracer(camera, kerr, blackHole)

    def test_ray(self):

        # Override the initial conditions of the raytracer
        self.rayTracer.override_initial_conditions(10,1.415,0,1.445,-0.659734)

        # Integrate the rays and collect the data
        self.rayTracer.collect_rays()

        # Read the Mathematica's ray data from the csv

        mathematica_data = np.genfromtxt('test_data/test_ray.csv',delimiter=',')
        mathematica_data = mathematica_data[:301]

        for row in range(0, self.rayTracer.imageRows):
            for col in range(0, self.rayTracer.imageCols):
                npt.assert_almost_equal(mathematica_data,
                                        self.rayTracer.rayData[row,col].T, decimal=5)



    def test_idempotency(self):
        # Override the initial conditions of the raytracer
        self.rayTracer.override_initial_conditions(10,1.415,0,1.445,-0.659734)

        # Integrate the rays and collect the data
        self.rayTracer.collect_rays()

        # Read the Mathematica's ray data from the csv

        first_run_data = self.rayTracer.rayData

        # Second run
        self.rayTracer.override_initial_conditions(10,1.415,0,1.445,-0.659734)
        self.rayTracer.collect_rays()
        second_run_data = self.rayTracer.rayData

        self.assertTrue( np.all(first_run_data == second_run_data) )


if __name__ == '__main__':
    # Run all the tests
    unittest.main()
