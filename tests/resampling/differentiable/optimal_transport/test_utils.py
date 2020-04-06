import tensorflow as tf

from filterflow.resampling.differentiable.regularized_transport.utils import squared_distances, softmin, cost


class TestUtils(tf.test.TestCase):
    def setUp(self):
        self.x = tf.random.uniform([5, 100, 2])
        self.y = tf.random.uniform([5, 100, 2])
        self.very_large_epsilon = tf.constant(100.)
        self.large_epsilon = tf.constant(0.1)
        self.small_epsilon = tf.constant(0.01)
        self.very_small_epsilon = tf.constant(1e-6)
        self.f = tf.zeros([5, 100], dtype=float)

    def test_squared_distances(self):
        distance = squared_distances(self.x, self.y)
        self.assertAllEqual(distance.shape, [5, 100, 100])
        self.assertAllClose(tf.reduce_mean(distance), 1 / 3, atol=1e-2)

    def test_cost(self):
        cost_matrix = cost(self.x, self.y)
        self.assertAllEqual(cost_matrix.shape, [5, 100, 100])
        self.assertAllClose(tf.reduce_mean(cost_matrix), 1 / 6, atol=1e-2)

    def test_softmin(self):
        cost_matrix = cost(self.x, self.x)
        very_small_eps_res = softmin(self.very_small_epsilon, cost_matrix, self.f)
        small_eps_res = softmin(self.small_epsilon, cost_matrix, self.f)
        large_eps_res = softmin(self.large_epsilon, cost_matrix, self.f)

        self.assertAllClose(very_small_eps_res, tf.zeros_like(very_small_eps_res), atol=1e-5)
        self.assertAllGreater(very_small_eps_res - small_eps_res, 0.)
        self.assertAllGreater(small_eps_res - large_eps_res, 0.)

        self.assertAllEqual(small_eps_res.shape, [5, 100])