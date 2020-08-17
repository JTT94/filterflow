import abc

import tensorflow as tf

from filterflow.base import State, Module


class ResamplingCriterionBase(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, state: State):
        """Flags which batches should be resampled

        :param state: State
            current state
        :return: mask of booleans
        :rtype tf.Tensor
        """


def neff(tensor, assume_normalized: bool, is_log: bool, threshold: float):
    if is_log:
        if assume_normalized:
            log_neff = -tf.reduce_logsumexp(2 * tensor, 1)
        else:
            log_neff = 2 * tf.reduce_logsumexp(tensor, 1) - tf.reduce_logsumexp(2 * tensor, 1)
        flag = log_neff <= tf.math.log(threshold)
        return flag, tf.exp(log_neff)
    else:
        if assume_normalized:
            neff = 1 / tf.reduce_sum(tensor ** 2, 1)
        else:
            neff = tf.reduce_sum(tensor, 1) ** 2 / tf.reduce_sum(tensor ** 2, 1)
        flag = neff <= threshold

        return flag, neff


class NeffCriterion(ResamplingCriterionBase):
    """
    Standard Neff criterion for resampling. If the neff of the state tensor falls below a certain threshold
    (either in relative or absolute terms) then the state will be flagged as needing resampling
    """

    def __init__(self, threshold, is_relative, on_log=True, assume_normalized=True, name='NeffCriterion'):
        super(NeffCriterion, self).__init__(name=name)
        self._threshold = threshold
        self._is_relative = is_relative
        self._on_log = on_log
        self._assume_normalized = assume_normalized

    def apply(self, state: State):
        """Flags which batches should be resampled

        :param state: State
            current state
        :return: mask of booleans, efficient sample size prior resampling
        :rtype tf.Tensor
        """
        threshold = self._threshold if not self._is_relative else tf.cast(state.n_particles, float) * self._threshold
        if self._on_log:
            return neff(state.log_weights, self._assume_normalized, self._on_log, threshold)
        else:
            return neff(state.weights, self._assume_normalized, self._on_log, threshold)


class AlwaysResample(ResamplingCriterionBase):

    def apply(self, state: State):
        return tf.ones(state.batch_size, dtype=tf.bool), tf.zeros(state.batch_size, dtype=tf.float32)


class NeverResample(ResamplingCriterionBase):

    def apply(self, state: State):
        return tf.zeros(state.batch_size, dtype=tf.bool), tf.zeros(state.batch_size, dtype=tf.float32)
