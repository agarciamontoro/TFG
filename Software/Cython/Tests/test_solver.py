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

        cython_data = geodesic_integrator.integrate_ray(r = 10.0, cam_theta = 1.415, cam_phi = 0,
                                         theta_cs = 1.445, phi_cs = -0.659734, a= 0.8,
                                         n_steps = 300)

        current_file_path = os.path.dirname(os.path.realpath(__file__))
        mathematica_data = np.genfromtxt(os.path.join(current_file_path,
                                                      'test_data/test_ray.csv')
                                         ,delimiter=',')
        mathematica_data = mathematica_data[:301]
        npt.assert_almost_equal(mathematica_data,
                                cython_data,
                                decimal=5)


if __name__ == '__main__':
    unittest.main()
