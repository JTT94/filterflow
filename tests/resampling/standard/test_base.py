import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.standard.base import _discrete_percentile_function, StandardResamplerBase


class TestBaseFunctions(tf.test.TestCase):
    def setUp(self):
        self.spacings = tf.constant([[0.25, 0.66, 0.9],
                                     [0.5, 0.6, 0.8]])
        self.n_particles = 3
        self.weights = tf.constant([[0.26, 0.5, 0.23999],
                                    [0.3, 0.31, 0.39]])
        self.log_weights = tf.math.log(self.weights)

        self.particles = tf.reshape(tf.linspace(0., 2 * 3 * 4 - 1., 2 * 3 * 4),
                                    [2, 3, 4])  # batch_size, n_particles, dimension
        self.flags = tf.constant([False, True])

    def test_discrete_percentile_function(self):
        indices_from_log = _discrete_percentile_function(self.spacings,
                                                         self.n_particles,
                                                         True,
                                                         None,
                                                         self.log_weights)

        indices_from_raw = _discrete_percentile_function(self.spacings,
                                                         self.n_particles,
                                                         False,
                                                         self.weights,
                                                         None)

        self.assertAllEqual(indices_from_log, indices_from_raw)
        self.assertAllEqual(indices_from_log, tf.constant([[0, 1, 2],
                                                           [1, 1, 2]]))


class TestStandardResamplerBase(tf.test.TestCase):
    class Resampler(StandardResamplerBase):
        @staticmethod
        def _get_spacings(n_particles, batch_size):
            return tf.constant([[0.33, 0.5, 0.6],
                                [0.25, 0.5, 0.75]])

    def setUp(self):
        self.n_particles = 3
        self.batch_size = 2

        self.state = State(2, 3, 4,
                           tf.reshape(tf.linspace(0., 2 * 3 * 4 - 1., 2 * 3 * 4),
                                      [2, 3, 4]),
                           tf.constant([[0.26, 0.5, 0.23999],
                                        [0.3, 0.31, 0.39]]),
                           tf.constant([[0.26, 0.5, 0.23999],
                                        [0.3, 0.31, 0.39]]),
                           tf.constant([0., 0.]))

        self.flags = tf.constant([True, False])

        self.resampler = self.Resampler('Resampler', False)

    def test_apply(self):
        resampled_state = self.resampler.apply(self.state, self.flags)
        self.assertAllEqual(resampled_state.particles.shape, self.state.particles.shape)
        self.assertAllClose(resampled_state.particles[0, 0], self.state.particles[0, 1])
        self.assertAllClose(resampled_state.particles[0, 1], self.state.particles[0, 1])
        self.assertAllClose(resampled_state.particles[0, 2], self.state.particles[0, 1])
