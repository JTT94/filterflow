import numpy as np
import ot
import tensorflow as tf

from filterflow.resampling.differentiable.regularized_transport.plan import transport, sinkhorn_potentials
from filterflow.utils import normalize


class TestSinkhorn(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_particles = 25
        batch_size = 3
        dimension = 2

        self.n_particles = tf.constant(n_particles)
        self.batch_size = tf.constant(batch_size)
        self.dimension = tf.constant(dimension)

        self.np_epsilon = 0.5
        self.epsilon = tf.constant(self.np_epsilon)

        self.threshold = tf.constant(1e-3)
        self.n_iter = tf.constant(100)

        self.np_x = np.random.uniform(-1., 1., [batch_size, n_particles, dimension]).astype(np.float32)
        self.x = tf.constant(self.np_x)

        degenerate_weights = np.random.uniform(0., 1., [batch_size, n_particles]).astype(np.float32)
        degenerate_weights /= degenerate_weights.sum(axis=1, keepdims=True)

        self.degenerate_weights = degenerate_weights
        self.degenerate_logw = tf.math.log(degenerate_weights)

        self.uniform_logw = tf.zeros_like(degenerate_weights) - tf.math.log(float(n_particles))

    def test_transport(self):
        T_scaled = transport(self.x, self.degenerate_logw, self.epsilon, 0.9, self.threshold,
                             self.n_iter, self.n_particles)

        scale_np_x = np.max(self.np_x[0]) - np.min(self.np_x[0])
        self.assertAllClose(tf.constant(self.degenerate_weights) * tf.cast(self.n_particles, float),
                            tf.reduce_sum(T_scaled, 1))

        self.assertAllClose(tf.reduce_sum(T_scaled, 2), tf.ones_like(self.degenerate_logw), atol=1e-3)

        self.assertAllClose(tf.reduce_sum(T_scaled, [1, 2]),
                            tf.cast(self.n_particles, float) * tf.ones([self.batch_size]), atol=1e-3)

        np_transport_matrix = ot.bregman.empirical_sinkhorn(self.np_x[0] / scale_np_x, self.np_x[0] / scale_np_x,
                                                            self.np_epsilon,
                                                            b=self.degenerate_weights[0])

        self.assertAllClose(T_scaled[0], np_transport_matrix * self.n_particles.numpy(),
                            atol=1e-4)

    def test_gradient_transport(self):
        @tf.function
        def fun_x(x):
            transport_matrix = transport(x, self.degenerate_logw, self.epsilon, tf.constant(0.75), self.threshold,
                                         self.n_iter, self.n_particles)
            return tf.math.reduce_std(tf.linalg.matmul(transport_matrix, x))

        @tf.function
        def fun_logw(logw):
            logw = normalize(logw, 1, self.n_particles)

            transport_matrix = transport(self.x, logw, self.epsilon, tf.constant(0.9),
                                         self.threshold, self.n_iter, self.n_particles)
            return tf.math.reduce_std(tf.linalg.matmul(transport_matrix, self.x))

        theoretical, numerical = tf.test.compute_gradient(fun_x, [self.x], delta=1e-4)
        self.assertAllClose(theoretical[0], numerical[0], atol=1e-2)

        theoretical, numerical = tf.test.compute_gradient(fun_logw, [self.degenerate_logw], delta=1e-4)
        self.assertAllClose(theoretical[0], numerical[0], atol=1e-2)
