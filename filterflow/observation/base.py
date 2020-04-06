import abc

import tensorflow as tf

from filterflow.base import State, Observation, Module


class ObservationModelBase(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def loglikelihood(self, state: State, observation: Observation):
        """Computes the loglikelihood of an observation given proposed particles
        :param state: State
            Proposed (predicted) state of the filter given State at t-1 and Observation
        :param observation: Observation
            User/Process given observation
        :return: a tensor of loglikelihoods for all particles
        :rtype: tf.Tensor
        """


class ObservationSampler(ObservationModelBase, metaclass=abc.ABCMeta):
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

    @abc.abstractmethod
    def sample(self, state: State):
        """Samples a new observation conditionally on latent state
        :param state: State
            State of the filter at t
        :return: observartion 
        :rtype: ObservationBase
        """