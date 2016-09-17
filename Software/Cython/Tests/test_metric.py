import unittest
import numpy as np
import numpy.testing as npt
import sys
import os
sys.path.append('../')
import metric

class TestMetric(unittest.TestCase):
    def test_mathematica_comparison(self):

        kerr_metric  = metric.metric(1.9,0.2,0.2,0.7)
        methematica_kerr_metric = np.array([[-0.06877807693,0,0,-0.02572840654],
                                            [0,13.60219981,0,0],
                                            [0,0,4.080659944,0],
                                            [-0.02572840654,0,0,0.1625358035]])

        # Check the nonzero components

        npt.assert_almost_equal( kerr_metric, methematica_kerr_metric)

    def test_inverse(self):

        kerr_metric  = metric.metric(1.9,0.2,0.2,0.7)
        kerr_inverse_metric  = metric.inverse_metric(1.9,0.2,0.2,0.7)
        numpy_inverse = np.linalg.inv(kerr_metric)

        # Check that the calculated inverse is equal to numpy's inverse

        npt.assert_almost_equal( kerr_inverse_metric, numpy_inverse)
if __name__ == '__main__':
    unittest.main()
