import unittest
import numpy as np
import numpy.testing as npt
import sys
import os
sys.path.append('../')
import geodesic_integrator

class Test_Solver(unittest.TestCase):

    def test_mathematica_comparison(self):
        """

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

         This properties generate a ray that curves around the   black hole and comes back.
         The initial conditions and black hole properties are    selected in a way that the
         ray is very unstable in the sense that a little         variation of this set-up makes
         the ray to end in the horizon. This is very good to     test numerical stability.
        """

        cython_data = geodesic_integrator.test_integrate_camera_ray(r = 10.0, cam_theta = 1.415, cam_phi = 0,
                                         theta_cs = 1.445, phi_cs = -0.659734, a= 0.8,
                                         causality = 0, n_steps = 300)

        current_file_path = os.path.dirname(os.path.realpath(__file__))
        mathematica_data = np.genfromtxt(os.path.join(current_file_path,
                                                      'test_data/test_ray.csv')
                                         ,delimiter=',')
        mathematica_data = mathematica_data[:301]
        npt.assert_almost_equal(mathematica_data,
                                cython_data,
                                decimal=5)

    def test_general_integrator(self):
        """
        This test if the general pourpose integrator coincides with the previous solution. The main difference
        is that the Mathematica ray (and the integrate_camera_ray) recieve parameters relative to the camera
        and the inclination of the ray and the general pourpose integrator recieves directly the momenta and the
        initial position of the ray. The momenta of the test_mathematica_comparison set up is:

            Black Hole spin = 0.8

            - r     = 10
            - theta = 1.415
            - phi   = 0

            - pr     = 0.873022
            - ptheta = 1.25474
            - pphi   = -6.02992

        """
        three_position = np.array([10.0, 1.415, 0.0])
        three_momenta  = np.array([0.87302214, 1.25474473, -6.02991845])

        general_integrator_ray = geodesic_integrator.integrate_ray( three_position, three_momenta,
                                        causality = 0, a = 0.8, x0 = 0.0, xend = -30.0, n_steps = 300)

        current_file_path = os.path.dirname(os.path.realpath(__file__))
        mathematica_data = np.genfromtxt(os.path.join(current_file_path,
                                         'test_data/general_ray.csv')
                                         ,delimiter=',')
        mathematica_data = mathematica_data[:301]
        npt.assert_almost_equal(mathematica_data, general_integrator_ray, decimal=5)



if __name__ == '__main__':
    unittest.main()
