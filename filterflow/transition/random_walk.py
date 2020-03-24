import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State, InputsBase
from filterflow.transition.base import TransitionModelBase


class RandomWalkModel(TransitionModelBase):
    def __init__(self, transition_matrix: tf.Tensor, noise: tfp.distributions.Distribution):
        self._transition_matrix = transition_matrix
        self._noise = noise

    def loglikelihood(self, prior_state: State, proposed_state: State, inputs: InputsBase):
        """Computes the loglikelihood of an observation given proposed particles
        :param prior_state: State
            State at t-1
        :param proposed_state: State
            Some proposed State for which we want the likelihood given previous state
        :param inputs: InputsBase
            Input for transition model
        :return: a tensor of loglikelihoods for all particles in proposed state
        :rtype: tf.Tensor
        """
        pushed_particles = tf.linalg.matvec(self._transition_matrix, prior_state.particles)
        diff = proposed_state.particles - pushed_particles
        return self._noise.log_prob(diff)

    def sample(self, state: State, inputs: InputsBase):
        """Samples a new proposed state conditionally on prior state and some inputs
        :param state: State
            State of the filter at t-1
        :param inputs: InputsBase
            Input for transition model
        :return: proposed State
        :rtype: State
        """
        pushed_particles = tf.linalg.matvec(self._transition_matrix, state.particles)
        batch_size = state.batch_size
        n_particles = state.n_particles
        return pushed_particles + self._noise.sample([batch_size, n_particles])
