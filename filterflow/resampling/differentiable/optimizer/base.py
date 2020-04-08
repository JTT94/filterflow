import abc

import tensorflow as tf


class OptimizerBase(tf.Module, metaclass=abc.ABCMeta):
    """Optimizer."""

    @abc.abstractmethod
    def __call__(self, log_w_x, w_x, x, log_w_y, w_y, y):
        """Needs a nice docstring

        :param log_w_x:
        :param w_x:
        :param x:
        :param log_w_y:
        :param w_y:
        :param y:
        :return: The optimized point cloud starting from x
        :rtype: tf.Tensor
        """
