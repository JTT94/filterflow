import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.differentiable.loss.regularized import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD
from filterflow.resampling.standard.systematic import SystematicResampler


class TestOptimizedPointCloud(tf.test.TestCase):

    def setUp(self):
        N = 10
        n_particles = tf.constant(N)
        dimension = tf.constant(2)
        batch_size = tf.constant(3)

        self.flags = tf.constant([True, False, True])

        weights = tf.random.uniform((batch_size, n_particles), dtype=float)
        weights = weights / tf.reduce_sum(weights, 1, keepdims=True)
        particles = tf.random.uniform((batch_size, n_particles, dimension), -1, 1)
        log_likelihoods = tf.zeros(batch_size, dtype=float)
        self.state = State(particles=particles, log_weights=tf.math.log(weights),
                           weights=weights, log_likelihoods=log_likelihoods,
                           ancestor_indices=None, resampling_correction=None)

        self.loss = SinkhornLoss(tf.constant(0.05))
        loss_optimizer = SGD(self.loss, lr=0.5, n_iter=10)
        intermediate_resampling = SystematicResampler(on_log=tf.constant(True))
        self.cloud_optimizer = OptimizedPointCloud(loss_optimizer, intermediate_resampling)

    def test_apply(self):
        optimized_states = self.cloud_optimizer.apply(self.state, self.flags)

        self.assertNotAllClose(self.state.particles[0], optimized_states.particles[0])
        self.assertAllClose(self.state.particles[1], optimized_states.particles[1])

        self.assertAllClose(optimized_states.log_weights[0], -tf.math.log([10.] * 10))

        optimized_loss = self.loss(optimized_states.log_weights, optimized_states.weights, optimized_states.particles,
                                   self.state.log_weights, self.state.weights, self.state.particles)

        non_optimized_loss = self.loss(optimized_states.log_weights, optimized_states.weights, self.state.particles,
                                       self.state.log_weights, self.state.weights, self.state.particles)

        self.assertAllLess(optimized_loss[self.flags] - non_optimized_loss[self.flags], 0.)
        self.assertAllEqual(optimized_loss[tf.logical_not(self.flags)] - non_optimized_loss[tf.logical_not(self.flags)],
                            [0.])
