import scipy.stats as st
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.transition.random_walk import RandomWalkModel


class MockState(object):
    def __init__(self, particles):
        self.particles = particles
        self.batch_size, self.n_particles, self.dimension = particles.shape.as_list()


class TestRandomWalkModel(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)
        self.particles = tf.zeros([4, 10000, 3])
        self.rv = tfp.distributions.MultivariateNormalDiag([0., 0., 0.], [1., 1., 1.])
        self.transition_matrix = tf.eye(3) * 0.
        self.model = RandomWalkModel(self.transition_matrix, self.rv)

    def test_loglikelihood(self):
        logprob_1 = self.model.loglikelihood(MockState(self.particles), MockState(self.particles), None)
        self.assertAllEqual(logprob_1.shape,
                            self.particles.shape.as_list()[:2])
        self.assertAllClose(tf.reduce_mean(logprob_1, 1), [3 * st.norm.logpdf(0.)] * 4, atol=1e-3)
        # TODO: non scipy dependent test

    def test_sample(self):
        sample = self.model.sample(MockState(self.particles), None)
        self.assertAllEqual(sample.shape,
                            self.particles.shape.as_list())
        self.assertAllClose(tf.math.reduce_std(sample, [1]), [[1.] * 3] * 4, atol=1e-2)
