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
        """
        This is automatically called by the unittest suite when the class is created.

        This set up assumes the following:

            Black Hole spin = 0.8
            Sensor Shape = 10 x 10

            Camera Properties:
                - r     = 10
                - theta = 1.415
                - phi   = 0

            Ray Properties:

                - ThetaCS = 1.445,
                - PhiCS   = -0.659734

        This properties generate a ray that curves around the black hole and comes back.
        The initial conditions and black hole properties are selected in a way that the
        ray is very unstable in the sense that a little variation of this set-up makes
        the ray to end in the horizon. This is very good to test numerical stability.
        """

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


        # Override the initial conditions of the raytracer using the
        self.rayTracer.override_initial_conditions(10,1.415,0,1.445,-0.659734)

    def test_mathematica_comparison(self):
        """
        This test integrates the initial conditions described in the setUp method
        in a 10 x 10 grid using steps of -0.1 and compares each ray (that are supossed
        to be equal) to the result of the integration using Mathematica's NDSolve engine
        with the following options:

            - WorkingPrecission = 30

        As the ray is integrated using RK45 solver, the ray coordinates in each step are
        supposed to coincide up to 5 decimal places.

        """

        # Reset the initial conditions of the raytracer
        self.rayTracer.override_initial_conditions(10,1.415,0,1.445,-0.659734)

        # Integrate the rays and collect the data
        self.rayTracer.collect_rays()

        # Read the Mathematica's ray data from the csv

        mathematica_data = np.genfromtxt('test_data/test_ray.csv',delimiter=',')
        mathematica_data = mathematica_data[:301]

        # Compare each pixel's ray coordinates up to 5 decimal places with
        # mathematica's result.

        # Sorry for the loop with the numpy array!!
        for row in range(0, self.rayTracer.imageRows):
            for col in range(0, self.rayTracer.imageCols):
                npt.assert_almost_equal(mathematica_data,
                                        self.rayTracer.rayData[row,col].T,
                                        decimal=5)



    def test_idempotency(self):
        """
        This test checks that the result of integrating the ray that has the initial
        conditions of the setUp method are the same no matter how many times we
        call the kernel (after reseting the initial_conditios).

        The pourpose of the test is to detect memory leaks and erros in the kernel.
        """

        # Reset the initial conditions of the raytracer
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
