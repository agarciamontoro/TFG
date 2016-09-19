import unittest
import numpy as np
import numpy.testing as npt
import sys
sys.path.append('../')
import metric
import geodesic_integrator

class TestMetric(unittest.TestCase):
    def test_mathematica_comparison(self):

        kerr_metric  = metric.metric(1.9,0.2,0.7)
        methematica_kerr_metric = np.array([[-0.06877807693,0,0,-0.02572840654],
                                            [0,13.60219981,0,0],
                                            [0,0,4.080659944,0],
                                            [-0.02572840654,0,0,0.1625358035]])

        # Check the nonzero components

        npt.assert_almost_equal( kerr_metric, methematica_kerr_metric)

    def test_inverse(self):

        kerr_metric  = metric.metric(1.9,0.2,0.7)
        kerr_inverse_metric  = metric.inverse_metric(1.9,0.2,0.7)
        numpy_inverse = np.linalg.inv(kerr_metric)

        # Check that the calculated inverse is equal to numpy's inverse

        npt.assert_almost_equal( kerr_inverse_metric, numpy_inverse)

    def test_Kretschmann(self):

        npt.assert_almost_equal( metric.kretschmann(4,1.12,0.77),0.01005111985 )

    def test_temporal_component_momentum(self):

        # Random data for 3-vector
        three_vec = np.array([3,.12,.45])

        # Test lightlike four vector
        v = np.zeros(4)
        v[0] = geodesic_integrator.calculate_temporal_component( three_vec,three_vec,.4,causality=0)
        v[1:] = three_vec[:]
        norm = np.dot(np.dot(metric.inverse_metric(three_vec[0],three_vec[1],.4),v),v)
        npt.assert_almost_equal( norm, 0)

        # Test timelike four vector
        v = np.zeros(4)
        v[0] = geodesic_integrator.calculate_temporal_component( three_vec,three_vec,a=.4,causality=-1)
        v[1:] = three_vec[:]
        norm = np.dot(np.dot(metric.inverse_metric(three_vec[0],three_vec[1],a=.4),v),v)
        npt.assert_almost_equal( norm, -1)



if __name__ == '__main__':
    unittest.main()
