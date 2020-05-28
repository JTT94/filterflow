import abc

import tensorflow as tf

from filterflow.base import State, Module


@tf.function
def resample(tensor: tf.Tensor, new_tensor: tf.Tensor, flags: tf.Tensor):
    ndim = len(tensor.shape)
    shape = [-1] + [1] * (ndim - 1)
    return tf.where(tf.reshape(flags, shape), new_tensor, tensor)


class ResamplerBase(Module, metaclass=abc.ABCMeta):
    """Abstract ResamplerBase."""
    DIFFERENTIABLE = False

    @abc.abstractmethod
    def apply(self, state: State, flags: tf.Tensor, seed=None):
        """ Resampling method

        :param state: State
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """


class NoResampling(ResamplerBase):
    DIFFERENTIABLE = True

    def apply(self, state: State, flags: tf.Tensor, seed=None):
        """ Resampling method

        :param state: State
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :param seed: tf.Tensor
            seed for resampling (if needed)
        :return: resampled state
        :rtype: State
        """
        return state
