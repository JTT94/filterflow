import tensorflow as tf

import abc

import tensorflow as tf

from filterflow.base import State, InputsBase


class TransitionModelBase(tf.Module, metaclass=abc.ABCMeta):
    @tf.function
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
        return self._loglikelihood(prior_state.particles, proposed_state.particles, inputs)

    @abc.abstractmethod
    @tf.function
    def _loglikelihood(self, prior_particles: tf.Tensor, proposed_particles: tf.Tensor, inputs: InputsBase):
        """User defined implementation
        :param prior_particles: tf.Tensor
            Prior Particles from State
        :param proposed_particles: tf.Tensor
            Proposed Particles
        :param inputs: InputsBase
            Input for transition model
        :return: loglikelihoods of the proposed particles conditional on prior ones and inputs
        """

    @tf.function
    def sample(self, state: State, inputs: InputsBase):
        """Samples a new proposed state conditionally on prior state and some inputs
        :param state: State
            State of the filter at t-1
        :param inputs: InputsBase
            Input for transition model
        :return: proposed State
        :rtype: State
        """
        particles = self._sample(state.particles, inputs)
        return State(state.n_particles, state.batch_size, state.dimension, particles, state.log_weights,
                     state.weights, state.log_likelihood, False)

    @abc.abstractmethod
    @tf.function
    def _sample(self, particles: tf.Tensor, inputs: InputsBase):
        """Samples a new proposed state conditionally on prior state and some inputs
        :param particles: tf.Tensor
             particles for State at t-1
        :param inputs: InputsBase
            Input for transition model
        :return: generated particles
        :rtype: tf.Tensor
        """

