import abc

import tensorflow as tf

from filterflow.base import State


class ResamplerBase(metaclass=abc.ABCMeta):
    """Abstract ResamplerBase."""

    @abc.abstractmethod
    @tf.function
    def apply(self, state: State):
        """ Resampling method

        :param state: State
            Particle filter state
        :return: resampled state
        :rtype: State
        """


class NoResampling(ResamplerBase):
    @tf.function
    def apply(self, state: State):
        """ Resampling method

        :param state: State
            Particle filter state
        :return: resampled state
        :rtype: State
        """
        return state
