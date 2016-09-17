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
        methematica_kerr_metric = np.array([[-0.06877807693,0,0,-0.05145681308],
                                            [0,13.60219981,0,0],
                                            [0,0,4.080659944,0],
                                            [-0.05145681308,0,0,0.1625358035]])

        # Check the nonzero components

        npt.assert_almost_equal( kerr_metric, methematica_kerr_metric)

if __name__ == '__main__':
    unittest.main()
