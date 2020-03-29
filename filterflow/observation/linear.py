import attr
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State, ObservationBase, ObservationSeries
from filterflow.observation.base import ObservationModelBase


@attr.s
class LinearObservation(ObservationBase):
    observation = attr.ib(converter=lambda x: tf.reshape(x, [1, 1, -1]))

class LinearObservationSeries(ObservationSeries):
    def __init__(self, dtype, dimension):
        self.observation_ta = tf.TensorArray(dtype, 
                                             size=0, dynamic_size=True, clear_after_read=False,
                                             element_shape = tf.TensorShape([1,1, dimension]))
        self.n_observations = 0
    
    def write(self, t, observation):
        self.observation_ta = self.observation_ta.write(t, observation.observation)
        self.n_observations = self.n_observations + 1
    
    def read(self, t):
        observation = self.observation_ta.read(t)
        observation = LinearObservation(observation)
        return observation

class LinearObservationModel(ObservationModelBase):
    def __init__(self, observation_matrix: tf.Tensor, error_rv: tfp.distributions.Distribution,
                 name='LinearObservationModel'):
        super(LinearObservationModel, self).__init__(name=name)
        self._observation_matrix = observation_matrix
        self._error_rv = error_rv

    def loglikelihood(self, state: State, observation: ObservationBase):
        """Computes the loglikelihood of an observation given proposed particles
        :param state: State
            Proposed (predicted) state of the filter given State at t-1 and Observation
        :param observation: ObservationBase
            User/Process given observation
        :return: a tensor of loglikelihoods for all particles
        :rtype: tf.Tensor
        """
        error = observation.observation - tf.linalg.matvec(self._observation_matrix, state.particles)
        return self._error_rv.log_prob(error)
