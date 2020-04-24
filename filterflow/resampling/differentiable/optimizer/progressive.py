import tensorflow as tf

from filterflow.resampling.differentiable.optimizer.base import OptimizerBase


@tf.function
def l_2_barycenter(w_1, w_2):
    pass

@tf.function
def sinkhorn_barycenter(w_1, w_2, epsilon):
    pass


class ProgressiveOptimizer(OptimizerBase):
    # TODO: docstring!
    def __init__(self, loss, n_iter_in, n_iter_out, lr, decay, grad_threshold=1e-4, name='ProgressiveOptimizer'):
        super(ProgressiveOptimizer, self).__init__(name=name)
        self._loss = loss
        self._n_iter_in = tf.cast(n_iter_in, tf.int32)
        self._n_iter_out = tf.cast(n_iter_out, tf.int32)
        self._lr = tf.cast(lr, float)
        self._decay = tf.cast(decay, float)
        self._grad_threshold = tf.cast(grad_threshold, float)

    def __call__(self, log_w_x, w_x, x, log_w_y, w_y, y):
        pass

