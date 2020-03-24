import scipy.stats as st
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.observation.linear import LinearObservationModel, LinearObservation


class MockState(object):
    def __init__(self, particles):
        self.particles = particles


class TestLinearObservationModel(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)
        self.particles = tf.zeros([3, 10, 3])
        self.rv = tfp.distributions.MultivariateNormalDiag([0., 0., 0.], [1., 1., 1.])
        self.observation_matrix = tf.eye(3)
        self.model = LinearObservationModel(self.observation_matrix, self.rv)

    def test_loglikelihood(self):
        logprob_1 = self.model.loglikelihood(MockState(self.particles),
                                             LinearObservation(tf.constant([-1., 0., 1.]), 3))
        self.assertAllEqual(logprob_1.shape,
                            self.particles.shape.as_list()[:2])

        self.assertAllClose(tf.reduce_mean(logprob_1, 1), [st.norm.logpdf(0.) + 2 * st.norm.logpdf(1.)] * 3)
        # TODO: non scipy dependent test
