import abc

import tensorflow as tf

from filterflow.base import State, Module


class ResamplerBase(Module, metaclass=abc.ABCMeta):
    """Abstract ResamplerBase."""

    @abc.abstractmethod
    def apply(self, state: State, flags: tf.Tensor):
        """ Resampling method

        :param state: State
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """


class NoResampling(ResamplerBase):

    def apply(self, state: State, flags: tf.Tensor):
        """ Resampling method

        :param state: State
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """
        return state
