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
                 name='SinkhornLoss'):
        """Constructor

        :param epsilon:
        :param symmetric:
        :param scaling:
        :param max_iter:
        :param convergence_threshold:
        """
        super(SinkhornLoss, self).__init__(name=name)
        assert not symmetric, "symmetric sinkhorn should be implemented, just not yet"

        self.symmetric = tf.cast(symmetric, bool)
        self.convergence_threshold = tf.cast(convergence_threshold, float)
        self.max_iter = tf.cast(max_iter, tf.dtypes.int32)
        self.epsilon = tf.cast(epsilon, float)
        self.scaling = tf.cast(scaling, float)

    def __call__(self, log_w_x, w_x, x, log_w_y, w_y, y):
        if not self.symmetric:
            a_y, b_x, _ = sinkhorn_potentials(log_w_x, x, log_w_y, y, self.epsilon, self.scaling,
                                              self.convergence_threshold, self.max_iter)
            return _scal(w_x, b_x) + _scal(w_y, a_y)
        else:
            raise ValueError('Symmetric is not supported yet')
