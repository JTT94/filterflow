import numpy as np
import tensorflow as tf

from filterflow.resampling.differentiable.optimal_transport.plan import solve_for_state, transport, \
    transport_from_potentials


class TestPlan(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)
        n_particles = 10
        batch_size = 2

        self.n_particles = tf.constant(n_particles)
        self.batch_size = tf.constant(batch_size)
        self.dimension = tf.constant(3)

        self.f = tf.zeros([self.batch_size, self.n_particles])
        self.g = tf.zeros([self.batch_size, self.n_particles])

        self.small_epsilon = tf.constant(1e-6)
        self.medium_epsilon = tf.constant(1e-2)
        self.large_epsilon = tf.constant(2e-2)
        self.very_large_epsilon = tf.constant(1.)

        self.threshold = tf.constant(1e-6)
        self.n_iter = tf.constant(100)

        self.x = tf.random.uniform([self.batch_size, self.n_particles, self.dimension], -1., 1., dtype=float)
        self.uniform_logw = tf.zeros([self.batch_size, self.n_particles], dtype=float) - np.log(n_particles)

        degenerate_weights = np.zeros([batch_size, n_particles], dtype=np.float32)
        degenerate_weights += 1e-10
        degenerate_weights[0, n_particles // 2] = 1.
        degenerate_weights[1, n_particles // 3] = 0.5
        degenerate_weights[1, 2 * n_particles // 3] = 0.5
        degenerate_weights /= degenerate_weights.sum(axis=1, keepdims=True)

        self.degenerate_logw = tf.math.log(degenerate_weights)

    def test_solve_for_state_uniform(self):
        res_small_eps = solve_for_state(self.x, self.uniform_logw, self.small_epsilon, self.threshold,
                                        self.n_iter, self.n_particles)

        res_medium_eps = solve_for_state(self.x, self.uniform_logw, self.medium_epsilon, self.threshold,
                                         self.n_iter, self.n_particles)

        res_large_eps = solve_for_state(self.x, self.uniform_logw, self.large_epsilon, self.threshold,
                                        self.n_iter, self.n_particles)

        res_f, res_g = res_small_eps

        self.assertAllClose(res_f, res_g, atol=1e-10)

        # The result is locally linear in epsilon
        self.assertAllClose(res_small_eps, self.small_epsilon / self.medium_epsilon * res_medium_eps,
                            atol=self.small_epsilon)

        self.assertAllClose(res_medium_eps, self.medium_epsilon / self.large_epsilon * res_large_eps,
                            atol=self.medium_epsilon)

    def test_solve_for_state_degenerate(self):
        res_small_eps = solve_for_state(self.x, self.degenerate_logw, self.small_epsilon, self.threshold,
                                        self.n_iter, self.n_particles)

        res_medium_eps = solve_for_state(self.x, self.degenerate_logw, self.medium_epsilon, self.threshold,
                                         self.n_iter, self.n_particles)

        res_large_eps = solve_for_state(self.x, self.degenerate_logw, self.large_epsilon, self.threshold,
                                        self.n_iter, self.n_particles)

        res_f, res_g = res_small_eps

        self.assertAllEqual(res_f.shape, [self.batch_size, self.n_particles])
        self.assertAllEqual(res_g.shape, [self.batch_size, self.n_particles])
        self.assertAllEqual(tf.argmax(res_f, 1), tf.argmin(res_g, 1))
        self.assertAllEqual(tf.argmax(res_f, 1), tf.argmax(self.degenerate_logw, 1))

        # The potentials are linear in epsilon in the first order
        self.assertAllClose(res_small_eps, self.small_epsilon / self.medium_epsilon * res_medium_eps,
                            atol=self.small_epsilon)
        self.assertAllClose(res_medium_eps, self.medium_epsilon / self.large_epsilon * res_large_eps,
                            atol=self.medium_epsilon)

    def test_transport_from_potentials(self):
        particles, log_weights = transport_from_potentials(self.x, self.f, self.g, self.small_epsilon,
                                                           self.degenerate_logw, self.n_particles)

        self.assertAllClose(log_weights, self.uniform_logw)
        self.assertAllClose(particles, self.x)

        particles, log_weights = transport_from_potentials(self.x, self.f, self.g, self.very_large_epsilon * 100.,
                                                           self.degenerate_logw, self.n_particles)

        self.assertAllClose(log_weights, self.uniform_logw)

        self.assertAllClose(particles - tf.reduce_mean(self.x, 1, keepdims=True), tf.zeros_like(particles), atol=1e-1)

    def test_transport_uniform(self):
        particles, log_weights = transport(self.x, self.uniform_logw, self.small_epsilon, self.threshold,
                                           self.n_particles, self.n_iter)

        self.assertAllClose(log_weights, self.uniform_logw)
        self.assertAllClose(particles, self.x)

        particles, log_weights = transport(self.x, self.uniform_logw, self.very_large_epsilon * 100., self.threshold,
                                           self.n_particles, self.n_iter)

        self.assertAllClose(log_weights, self.uniform_logw)
        self.assertAllClose(particles - tf.reduce_mean(self.x, 1, keepdims=True), tf.zeros_like(particles), atol=1e-1)

    def test_transport_degenerate(self):
        particles, log_weights = transport(self.x, self.degenerate_logw, self.small_epsilon, self.threshold,
                                           self.n_particles, self.n_iter)

        self.assertAllClose(log_weights, self.uniform_logw)
        self.assertAllClose(particles, self.x)

        # TODO: any higher epsilon just returns NaNs. To be checked properly.
        particles, log_weights = transport(self.x, self.degenerate_logw, self.very_large_epsilon, self.threshold,
                                           self.n_particles, self.n_iter)

        diff = particles - tf.reduce_sum(self.x * tf.expand_dims(tf.math.exp(self.degenerate_logw), 2), 1,
                                         keepdims=True)

        indices = tf.reduce_sum(tf.abs(diff), 2) > 1e-1
        self.assertAllClose(log_weights, self.uniform_logw)
        self.assertAllEqual(indices, tf.math.exp(self.degenerate_logw) > 1/tf.cast(self.n_particles, float))
