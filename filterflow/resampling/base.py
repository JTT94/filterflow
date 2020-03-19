import abc

import tensorflow as tf

from filterflow.base import State


class Resampler(metaclass=abc.ABCMeta):
    """Abstract Resampler."""

    @abc.abstractmethod
    @tf.function
    def apply(self, state: State):
        """ Resampling method

        :param state: State
            Particle filter state
        :return: resampled state
        :rtype: State
        """


class NoResampling(Resampler):
    @tf.function
    def apply(self, state: State):
        """ Resampling method

        :param state: State
            Particle filter state
        :return: resampled state
        :rtype: State
        """
        return state
