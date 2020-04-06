import numpy as np
import ot
import tensorflow as tf

from filterflow.resampling.differentiable.regularized_transport.plan import transport


class TestSinkhorn(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_particles = 5
        batch_size = 1
        dimension = 3

        self.n_particles = tf.constant(n_particles)
        self.batch_size = tf.constant(batch_size)
        self.dimension = tf.constant(dimension)

        self.np_epsilon = 5e-2
        self.epsilon = tf.constant(self.np_epsilon)

        self.threshold = tf.constant(1e-6)
        self.n_iter = tf.constant(1000)

        self.np_x = np.random.uniform(-1., 1., [batch_size, n_particles, dimension]).astype(np.float32)
        self.x = tf.constant(self.np_x)

        degenerate_weights = np.zeros([batch_size, n_particles], dtype=np.float32)
        degenerate_weights += 1e-2
        degenerate_weights[0, n_particles // 2] = 1.
        degenerate_weights /= degenerate_weights.sum(axis=1, keepdims=True)
        self.degenerate_weights = degenerate_weights
        self.degenerate_logw = tf.math.log(degenerate_weights)

        self.uniform_logw = tf.zeros_like(degenerate_weights) - tf.math.log(float(n_particles))

    def test_transport(self):
        T_scaled, total_iter_scaled = transport(self.x, self.degenerate_logw, self.epsilon, 0.85, self.threshold,
                                                self.n_iter,
                                                self.n_particles)

        T_simple, total_iter_simple = transport(self.x, self.degenerate_logw, self.epsilon, 1., self.threshold,
                                                self.n_iter,
                                                self.n_particles)

        self.assertGreater(total_iter_simple, total_iter_scaled)

        self.assertAllClose(tf.constant(self.degenerate_weights) * tf.cast(self.n_particles, float),
                            tf.reduce_sum(T_scaled, 1))

        self.assertAllClose(tf.reduce_sum(T_scaled, 2), tf.ones_like(self.degenerate_logw), atol=0.05)

        self.assertAllClose(tf.reduce_sum(T_scaled, [1, 2]),
                            tf.cast(self.n_particles, float) * tf.ones([self.batch_size]), atol=1e-6)

        np_transport_matrix = ot.bregman.empirical_sinkhorn(self.np_x[0], self.np_x[0], self.epsilon,
                                                            b=self.degenerate_weights[0])

        self.assertAllClose(T_scaled[0], np_transport_matrix * self.n_particles.numpy(), atol=1e-1)

    def test_gradient(self):
        @tf.function
        def fun(x):
            transport_matrix, _ = transport(x, self.degenerate_logw, self.epsilon, tf.constant(0.9), tf.constant(1e-5),
                                            tf.constant(300), self.n_particles)
            return tf.math.reduce_std(tf.einsum('ijk,ikm->ijm', transport_matrix, x))

        theoretical, numerical = tf.test.compute_gradient(fun, [self.x], delta=1e-3)

        # Logic is not sensitive to translations.
        self.assertLess(tf.abs(tf.reduce_sum(theoretical[0])), tf.abs(tf.reduce_sum(numerical[0])))


