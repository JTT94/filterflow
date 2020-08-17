import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State, StateSeries
from filterflow.observation.linear import LinearObservationModel
from filterflow.proposal import BootstrapProposalModel
from filterflow.resampling.criterion import NeffCriterion
from filterflow.resampling.standard.systematic import SystematicResampler
from filterflow.smc import SMC
from filterflow.transition.random_walk import RandomWalkModel


class TestSMC(tf.test.TestCase):
    def setUp(self):
        N = 10
        n_particles = tf.constant(N)
        dimension = tf.constant(1)
        batch_size = tf.constant(4)

        weights = tf.ones((batch_size, n_particles), dtype=float) / tf.cast(n_particles, float)
        initial_particles = tf.random.uniform((batch_size, n_particles, dimension), -1, 1)
        log_likelihoods = tf.zeros((batch_size), dtype=float)
        self.initial_state = State(particles=initial_particles, log_weights=tf.math.log(weights), weights=weights,
                                   log_likelihoods=log_likelihoods, ancestor_indices=None, resampling_correction=None)

        error_variance = tf.constant([0.5], dtype=tf.float32)
        error_rv = tfp.distributions.MultivariateNormalDiag(tf.constant([0.]),
                                                            error_variance)

        noise_variance = tf.constant([0.5])
        noise_rv = tfp.distributions.MultivariateNormalDiag(tf.constant([0.]),
                                                            noise_variance)

        observation_model = LinearObservationModel(tf.constant([[1.]]), error_rv)

        transition_matrix = tf.constant([[1.]])
        transition_model = RandomWalkModel(transition_matrix, noise_rv)

        bootstrap = BootstrapProposalModel(transition_model)
        resampling_criterion = NeffCriterion(tf.constant(0.5), is_relative=tf.constant(True))
        systematic_resampling_method = SystematicResampler()

        self.bootstrap_filter = SMC(observation_model, transition_model, bootstrap, resampling_criterion,
                                    systematic_resampling_method)

        # TODO: Let's change this using an instance of StateSpaceModel
        self.n = 100
        observation = np.array([[[0.]]]).astype(np.float32)
        observations = []
        for _ in range(self.n):
            observations.append(observation)
            observation = observation + np.random.normal(0., 1., [1, 1, 1])
        self.observation_dataset = tf.data.Dataset.from_tensor_slices(observations)

    def test_call(self):
        final_state = self.bootstrap_filter(self.initial_state, self.observation_dataset, self.n,
                                            return_final=True)
        self.assertIsInstance(final_state, State)

        all_states = self.bootstrap_filter(self.initial_state, self.observation_dataset, self.n,
                                           return_final=False)
        self.assertIsInstance(all_states, StateSeries)
