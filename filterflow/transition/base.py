import abc

import tensorflow as tf

from filterflow.base import State, Module


class TransitionModelBase(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def loglikelihood(self, prior_state: State, proposed_state: State, inputs: tf.Tensor):
        """Computes the loglikelihood of an observation given proposed particles
        :param prior_state: State
            State at t-1
        :param proposed_state: State
            Some proposed State for which we want the likelihood given previous state
        :param inputs: tf.Tensor
            Input for transition model
        :return: a tensor of loglikelihoods for all particles in proposed state
        :rtype: tf.Tensor
        """

    @abc.abstractmethod
    def sample(self, state: State, inputs: tf.Tensor, seed=None):
        """Samples a new proposed state conditionally on prior state and some inputs
        :param state: State
            State of the filter at t-1
        :param inputs: tf.Tensor
            Input for transition model
        :param seed: tf.Tensor
            Input for distribution
        :return: proposed State
        :rtype: State
        """
