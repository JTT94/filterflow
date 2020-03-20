import abc
import math

import tensorflow as tf

from filterflow.base import State


class ResamplingCriterionBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, state: State):
        """Flags which batches of the state should be resampled"""


class NeffCriterion(ResamplingCriterionBase):
    """
    Standard Neff criterion for resampling. If the neff of the state weights falls below a certain threshold
    (either in relative or absolute terms) then the state will be flagged as needing resampling
    """

    def __init__(self, threshold, is_relative, on_log=True, assume_normalized=True):
        self._threshold = threshold
        self._is_relative = is_relative
        self._on_log = on_log
        self._assume_normalized = assume_normalized

    @tf.function
    def apply(self, state: State):
        """See base class"""
        if self._is_relative:
            threshold = self._threshold * state.n_particles
        else:
            threshold = self._threshold

        if self._on_log:
            if self._assume_normalized:
                log_neff = -tf.reduce_logsumexp(state.log_weights, 0)
            else:
                log_neff = 2 * tf.reduce_logsumexp(state.log_weights, 0) - tf.reduce_logsumexp(2 * state.log_weights, 0)
            return log_neff <= math.log(threshold)
        else:
            if self._assume_normalized:
                neff = 1 / tf.reduce_sum(state.weights ** 2, 0)
            else:
                neff = tf.reduce_sum(state.weights, 0) ** 2 / tf.reduce_sum(state.weights ** 2, 0)
        return neff <= threshold
