import attr
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.observation.linear import LinearObservationModel
from filterflow.proposal import ProposalModelBase
from filterflow.smc import SMC
from filterflow.transition.random_walk import RandomWalkModel

tfd = tfp.distributions


class LearnableProposalModel(ProposalModelBase):
    def __init__(self, transition_matrix, name='LearnableProposalModel'):
        super(LearnableProposalModel, self).__init__(name=name)
        self._transition_matrix = transition_matrix
        self._standard_noise = tfp.distributions.MultivariateNormalDiag(tf.zeros(transition_matrix.shape[0]),
                                                                        tf.ones(transition_matrix.shape[0]))

    def propose(self, state: State, inputs, _observation: tf.Tensor):
        """See base class"""
        mu_t, beta_t, sigma_t = inputs

        transition_matrix = tf.linalg.matmul(tf.linalg.diag(beta_t), self._transition_matrix)

        pushed_particles = tf.reshape(mu_t, [1, 1, -1]) + tf.linalg.matvec(transition_matrix, state.particles)

        scale = tfp.bijectors.ScaleMatvecDiag(sigma_t)
        scaled_rv = tfd.TransformedDistribution(self._standard_noise, bijector=scale)
        proposed_particles = pushed_particles + scaled_rv.sample([state.batch_size, state.n_particles])
        return attr.evolve(state, particles=proposed_particles)

    def loglikelihood(self, proposed_state: State, state: State, inputs: tf.Tensor, observation: tf.Tensor):
        """Interface method for particle proposal
        :param proposed_state: State
            proposed state
        :param state: State
            previous particle filter state
        :param inputs: tf.Tensor
            Control variables (time elapsed, some environment variables, etc)
        :param observation: tf.Tensor
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: tf.Tensor
        """
        mu_t, beta_t, sigma_t = inputs
        transition_matrix = tf.linalg.matmul(tf.linalg.diag(beta_t), self._transition_matrix)
        pushed_particles = tf.reshape(mu_t, [1, 1, -1]) + tf.linalg.matvec(transition_matrix, state.particles)

        scale = tfp.bijectors.ScaleMatvecDiag(sigma_t)
        scaled_rv = tfd.TransformedDistribution(self._standard_noise, bijector=scale)

        diff = (pushed_particles - proposed_state.particles)
        return scaled_rv.log_prob(diff)


def make_filter(observation_matrix, transition_matrix, observation_error_chol, transition_noise_chol, resampling_method,
                resampling_criterion, observation_error_bias=None, transition_noise_bias=None):
    dy, dx = observation_matrix

    if observation_error_bias is None:
        observation_error_bias = tf.zeros(dy)
    if transition_noise_bias is None:
        transition_noise_bias = tf.zeros(dx)

    observation_error = tfp.distributions.MultivariateNormalTriL(observation_error_bias, observation_error_chol)
    observation_model = LinearObservationModel(observation_matrix, observation_error)

    transition_noise = tfp.distributions.MultivariateNormalTriL(transition_noise_bias, transition_noise_chol)
    transition_model = RandomWalkModel(transition_matrix, transition_noise)
    proposal_model = LearnableProposalModel(transition_matrix)

    return SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)
