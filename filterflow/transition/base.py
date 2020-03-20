import abc

import tensorflow as tf

from filterflow.base import State, InputsBase


class TransitionModelBase(tf.Module, metaclass=abc.ABCMeta):
    def __init__(self, batch_size, n_particles, name='ObservationModelBase'):
        super(TransitionModelBase, self).__init__(name=name)
        self._batch_size = batch_size
        self._n_particles = n_particles

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_particles(self):
        return self._n_particles

    @abc.abstractmethod

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

    @abc.abstractmethod

    def sample(self, state: State, inputs: InputsBase):
        """Samples a new proposed state conditionally on prior state and some inputs
        :param state: State
            State of the filter at t-1
        :param inputs: InputsBase
            Input for transition model
        :return: proposed State
        :rtype: State
        """
