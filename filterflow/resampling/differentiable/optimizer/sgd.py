import tensorflow as tf

from filterflow.resampling.differentiable.optimizer.base import OptimizerBase


class SGD(OptimizerBase):
    """Optimizer."""

    def __init__(self, loss, lr=0.1, n_iter=50, name='SGD'):
        """Needs a nice

        :param loss:
        :param lr:
        :param n_iter:
        :param name:
        """
        super(SGD, self).__init__(name=name)
        self.loss = loss
        self.lr = lr
        self.n_iter = n_iter

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

        # TODO: this will not work exactly the right way with batch_size > 1:
        #  the gradient will be applied on the sum of the losses instead of being applied batch by batch.
        #  Might need to unstack z and then stack the gradient back. However this is enough for the time being.
        def body(i, z):
            with tf.GradientTape() as tape:
                tape.watch(z)
                loss = self.loss(log_w_x, w_x, z, log_w_y, w_y, y)
            grad = tape.gradient(loss, z)[0]
            return i + 1, z - self.lr * grad

        def cond(i, _z):
            return i < self.n_iter

        i_0 = tf.constant(0)
        _, res = tf.while_loop(cond, body, [i_0, x])
        return res
