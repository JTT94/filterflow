import abc

import tensorflow as tf

from filterflow.base import State, Module


class ObservationModelBase(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def loglikelihood(self, state: State, observation: tf.Tensor):
        """Computes the loglikelihood of an observation given proposed particles
        :param state: State
            Proposed (predicted) state of the filter given State at t-1 and Observation
        :param observation: tf.Tensor
            User/Process given observation
        :return: a tensor of loglikelihoods for all particles
        :rtype: tf.Tensor
        """


