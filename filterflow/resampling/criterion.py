import abc
import math
from typing import List

import tensorflow as tf

from filterflow.base import State


class ResamplingCriterionBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, states: List[State]):
        """Flags which states should be resampled

        :param states: List[State]
            current states
        :return: mask of booleans
        :rtype tf.Tensor
        """


@tf.function
def _neff(tensor, assume_normalized: bool, is_log: bool, threshold: float) -> tf.Tensor:
    if is_log:
        if assume_normalized:
            log_neff = -tf.reduce_logsumexp(2 * tensor, 1)
        else:
            log_neff = 2 * tf.reduce_logsumexp(tensor, 1) - tf.reduce_logsumexp(2 * tensor, 1)
        return log_neff <= math.log(threshold)
    else:
        if assume_normalized:
            neff = 1 / tf.reduce_sum(tensor ** 2, 1)
        else:
            neff = tf.reduce_sum(tensor, 1) ** 2 / tf.reduce_sum(tensor ** 2, 1)
    return neff <= threshold


class NeffCriterion(ResamplingCriterionBase):
    """
    Standard Neff criterion for resampling. If the neff of the state tensor falls below a certain threshold
    (either in relative or absolute terms) then the state will be flagged as needing resampling
    """

    def __init__(self, threshold, n_particles, is_relative, on_log=True, assume_normalized=True):
        self._threshold = threshold * n_particles if is_relative else threshold
        self._on_log = on_log
        self._assume_normalized = assume_normalized

    def apply(self, states: List[State]):
        """Flags which states should be resampled

        :param states: List[State]
            current states
        :return: mask of booleans
        :rtype tf.Tensor
        """
        if self._on_log:
            log_weights = tf.stack([state.log_weights for state in states], 0)
            return _neff(log_weights, self._assume_normalized, self._on_log, self._threshold)
        else:
            weights = tf.stack([state.weights for state in states], 0)
            return _neff(weights, self._assume_normalized, self._on_log, self._threshold)
