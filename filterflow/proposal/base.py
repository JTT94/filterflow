import abc

import tensorflow as tf

from filterflow.base import State, Module


class ProposalModelBase(Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def propose(self, state: State, inputs: tf.Tensor, observation: tf.Tensor, seed=None):
        """Interface method for particle proposal

        :param state: State
            previous particle filter state
        :param inputs: tf.Tensor
            Control variables (time elapsed, some environment variables, etc)
        :param observation: tf.Tensor
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: State
        """

    @abc.abstractmethod
    def loglikelihood(self, proposed_state: State, state: State, inputs: tf.Tensor, observation: tf.Tensor):
        """Interface method for particle proposal
        :param proposed_state: State
            proposed state
        :param state: State
            previous particle filter state
        :param inputs: tf.Tensor
            Control variables (time elapsed, some environment variables, etc)
        :param observation: tf.Tensor
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: tf.Tensor
        """
