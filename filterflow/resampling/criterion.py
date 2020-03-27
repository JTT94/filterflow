import abc
import math

import tensorflow as tf

from filterflow.base import State


class ResamplingCriterionBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, state: State):
        """Flags which batches should be resampled

        :param state: State
            current state
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

    def __init__(self, threshold, is_relative, on_log=True, assume_normalized=True):
        self._threshold = threshold
        self._is_relative = is_relative
        self._on_log = on_log
        self._assume_normalized = assume_normalized

    def apply(self, state: State):
        """Flags which batches should be resampled

        :param state: State
            current state
        :return: mask of booleans
        :rtype tf.Tensor
        """
        threshold = self._threshold if not self._is_relative else state.n_particles * self._threshold
        if self._on_log:
            return _neff(state.log_weights, self._assume_normalized, self._on_log, threshold)
        else:
            return _neff(state.weights, self._assume_normalized, self._on_log, threshold)

class AlwaysResample(ResamplingCriterionBase):

    def apply(self, state: State):
        return True
