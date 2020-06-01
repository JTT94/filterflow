import tensorflow as tf

from filterflow.resampling.differentiable.loss.base import Loss
from filterflow.resampling.differentiable.regularized_transport.sinkhorn import sinkhorn_potentials

__all__ = ['SinkhornLoss']


@tf.function
def _scal(weight, potential):
    return tf.einsum('ij,ij->i', weight, potential)


class SinkhornLoss(Loss):
    # TODO: document
    def __init__(self, epsilon, symmetric=False, scaling=0.75, max_iter=50, convergence_threshold=1e-4,
                 name='SinkhornLoss', **_kwargs):
        """Constructor

        :param epsilon:
        :param symmetric:
        :param scaling:
        :param max_iter:
        :param convergence_threshold:
        """
        super(SinkhornLoss, self).__init__(name=name)

        self.symmetric = symmetric
        self.convergence_threshold = tf.cast(convergence_threshold, float)
        self.max_iter = tf.cast(max_iter, tf.dtypes.int32)
        self.epsilon = tf.cast(epsilon, float)
        self.scaling = tf.cast(scaling, float)

    def __call__(self, log_w_x, w_x, x, log_w_y, w_y, y):
        if not self.symmetric:
            a_y, b_x, _, _, _ = sinkhorn_potentials(log_w_x, x, log_w_y, y, self.epsilon, self.scaling,
                                                    self.convergence_threshold, self.max_iter)
            return _scal(w_x, b_x) + _scal(w_y, a_y)
        else:
            a_y, b_x, a_x, b_y, _ = sinkhorn_potentials(log_w_x, x, log_w_y, y, self.epsilon, self.scaling,
                                                        self.convergence_threshold, self.max_iter)
            return _scal(w_x, b_x - a_x) + _scal(w_y, a_y - b_y)
