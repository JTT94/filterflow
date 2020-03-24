import abc
from typing import List

import tensorflow as tf

from filterflow.base import State


class ResamplerBase(metaclass=abc.ABCMeta):
    """Abstract ResamplerBase."""

    @abc.abstractmethod
    def apply(self, states: List[State], flags: tf.Tensor):
        """ Resampling method

        :param states: List[State]
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: list of resampled states
        :rtype: List[State]
        """


class NoResampling(ResamplerBase):

    def apply(self, states: List[State], flags: tf.Tensor):
        """ Resampling method

        :param states: State
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: list of resampled states
        :rtype: List[State]
        """
        return states
