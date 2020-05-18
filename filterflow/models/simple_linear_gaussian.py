import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.observation.linear import LinearObservationModel
from filterflow.proposal import BootstrapProposalModel
from filterflow.smc import SMC
from filterflow.transition.random_walk import RandomWalkModel


def make_filter(observation_matrix, transition_matrix, observation_error_chol, transition_noise_chol, resampling_method,
                resampling_criterion, observation_error_bias=None, transition_noise_bias=None):
    dy, dx = observation_matrix.shape

    if observation_error_bias is None:
        observation_error_bias = tf.zeros(dy)
    if transition_noise_bias is None:
        transition_noise_bias = tf.zeros(dx)

    observation_error = tfp.distributions.MultivariateNormalTriL(observation_error_bias, observation_error_chol)
    observation_model = LinearObservationModel(observation_matrix, observation_error)

    transition_noise = tfp.distributions.MultivariateNormalTriL(transition_noise_bias, transition_noise_chol)
    transition_model = RandomWalkModel(transition_matrix, transition_noise)
    proposal_model = BootstrapProposalModel(transition_model)

    return SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)
