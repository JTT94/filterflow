import tensorflow as tf
import ot

from filterflow.resampling.differentiable.optimal_transport.utils import squared_distances, softmin

class TestUtils(tf.test.TestCase):
    def setUp(self):
        self.x = tf.random.uniform([5, 1000, 2])
        self.y = tf.random.uniform([5, 1000, 2])

    def test_squared_distances(self):
        distance = squared_distances(self.x, self.y)
        self.assertAllEqual(distance.shape, [5, 1000, 1000])
        self.assertAllClose(tf.reduce_mean(distance), 1/3, atol=1e-2)


