import abc

import tensorflow as tf

from filterflow.base import State, ObservationBase


class ObservationModelBase(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def loglikelihood(self, state: State, observation: ObservationBase):
        """Computes the loglikelihood of an observation given proposed particles
        :param state: State
            Proposed (predicted) state of the filter given State at t-1 and Observation
        :param observation: ObservationBase
            User/Process given observation
        :return: a tensor of loglikelihoods for all particles
        :rtype: tf.Tensor
        """
