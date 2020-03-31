import abc

import tensorflow as tf

from filterflow.base import State, Module


@tf.function
def resample(particles: tf.Tensor, new_particles: tf.Tensor, weights: tf.Tensor, new_weights: tf.Tensor,
             log_weights: tf.Tensor, new_log_weights: tf.Tensor, flags: tf.Tensor):
    particles = tf.where(tf.reshape(flags, [-1, 1, 1]),
                         new_particles,
                         particles)

    weights = tf.where(tf.reshape(flags, [-1, 1]),
                       new_weights,
                       weights)

    log_weights = tf.where(tf.reshape(flags, [-1, 1]),
                           new_log_weights,
                           log_weights)

    return particles, weights, log_weights


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
