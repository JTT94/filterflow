import abc

import tensorflow as tf

__all__ = ['Loss']


class Loss(tf.Module, metaclass=abc.ABCMeta):
    """A loss between two weighted point clouds."""

    @abc.abstractmethod
    def __call__(self, log_w_x, w_x, x, log_w_y, w_y, y):
        """Needs a nice docstring

        :param log_w_x:
        :param w_x:
        :param x:
        :param log_w_y:
        :param w_y:
        :param y:
        :return: The loss per batch between the point clouds
        :rtype: tf.Tensor
        """
