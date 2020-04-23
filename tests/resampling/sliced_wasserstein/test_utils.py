import tensorflow as tf
from scipy.stats.stats import _cdf_distance as st_cdf_distance

from filterflow.resampling.differentiable.loss.sliced_wasserstein import _cdf_distance


class TestUtils(tf.test.TestCase):
    def setUp(self):
        import numpy as np
        self.x = np.random.normal(0., 1., [1, 100])
        self.y = np.random.normal(0., 1., [1, 100])
        self.w_x = np.random.uniform(0., 1., [1, 100])
        self.w_y = np.random.uniform(0., 1., [1, 100])

        self.w_x /= self.w_x.sum(1, keepdims=True)
        self.w_y /= self.w_y.sum(1, keepdims=True)

    def test_cdf_distance(self):
        tf_res = _cdf_distance(self.x, self.y, self.w_x, self.w_y)
        sc_res = st_cdf_distance(2, self.x[0], self.y[0], self.w_x[0], self.w_y[0])
        self.assertAllClose(tf_res[0], sc_res)
