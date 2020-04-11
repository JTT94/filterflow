import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.observation.base import ObservationModelBase, ObservationSampler


class LinearObservationModel(ObservationModelBase):
    def __init__(self, observation_matrix: tf.Tensor, error_rv: tfp.distributions.Distribution,
                 name='LinearObservationModel'):
        super(LinearObservationModel, self).__init__(name=name)
        self._observation_matrix = observation_matrix
        self._error_rv = error_rv

    def loglikelihood(self, state: State, observation: tf.Tensor):
        """Computes the loglikelihood of an observation given proposed particles
        :param state: State
            Proposed (predicted) state of the filter given State at t-1 and Observation
        :param observation: tf.Tensor
            User/Process given observation
        :return: a tensor of loglikelihoods for all particles
        :rtype: tf.Tensor
        """
        error = observation - tf.linalg.matvec(self._observation_matrix, state.particles)
        return self._error_rv.log_prob(error)


class LinearObservationSampler(LinearObservationModel, ObservationSampler):

    def __init__(self, observation_matrix: tf.Tensor, error_rv: tfp.distributions.Distribution,
                 name="LinearObservationSampler"):
        super(LinearObservationSampler, self).__init__(observation_matrix=observation_matrix,
                                                       error_rv=error_rv,
                                                       name=name)

    def sample(self, state: State, inputs: tf.Tensor = tf.constant(0.)):
        """Samples a new proposed state conditionally on prior state and some inputs
        :param state: State
            State of the filter at t-1
        :param inputs: tf.Tensor
            Input for transition model
        :return: sampled Observation
        :rtype: Observation
        """
        pushed_particles = tf.linalg.matvec(self._observation_matrix, state.particles)
        observation = pushed_particles + self._error_rv.sample([state.batch_size, state.n_particles])
        return observation
