import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State, Observation
from filterflow.observation.base import ObservationModelBase


class LinearObservationModel(ObservationModelBase):
    def __init__(self, observation_matrix: tf.Tensor, error_rv: tfp.distributions.Distribution,
                 name='LinearObservationModel'):
        super(LinearObservationModel, self).__init__(name=name)
        self._observation_matrix = observation_matrix
        self._error_rv = error_rv

    def loglikelihood(self, state: State, observation: Observation):
        """Computes the loglikelihood of an observation given proposed particles
        :param state: State
            Proposed (predicted) state of the filter given State at t-1 and Observation
        :param observation: Observation
            User/Process given observation
        :return: a tensor of loglikelihoods for all particles
        :rtype: tf.Tensor
        """
        error = observation.observation - tf.linalg.matvec(self._observation_matrix, state.particles)
        return self._error_rv.log_prob(error)
