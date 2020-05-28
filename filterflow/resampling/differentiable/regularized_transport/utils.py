import tensorflow as tf


def _fillna(tensor):
    mask = tf.math.is_finite(tensor)
    return tf.where(mask, tensor, tf.zeros_like(tensor))


@tf.function
def diameter(x, y):
    diameter_x = tf.reduce_max(tf.math.reduce_std(x, -1), -1)
    diameter_y = tf.reduce_max(tf.math.reduce_std(y, -1), -1)

    return tf.maximum(diameter_x, diameter_y)


@tf.function
def max_min(x, y):
    max_max = tf.maximum(tf.math.reduce_max(x, [1, 2]), tf.math.reduce_max(y, [1, 2]))
    min_min = tf.minimum(tf.math.reduce_min(x, [1, 2]), tf.math.reduce_min(y, [1, 2]))

    return max_max - min_min


def softmin(epsilon: tf.Tensor, cost_matrix: tf.Tensor, f: tf.Tensor) -> tf.Tensor:
    """Implementation of softmin function

    :param epsilon: float
        regularisation parameter
    :param cost_matrix:
    :param f:
    :return:
    """
    n = cost_matrix.shape[1]
    b = cost_matrix.shape[0]

    f_ = tf.reshape(f, (b, 1, n))
    temp_val = f_ - cost_matrix / tf.reshape(epsilon, (-1, 1, 1))
    log_sum_exp = tf.reduce_logsumexp(temp_val, axis=2)
    res = -tf.reshape(epsilon, (-1, 1)) * log_sum_exp

    return res


@tf.function
def squared_distances(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """ Computes the square distance matrix on the last dimension between two tensors:

    :param x: tf.Tensor[B, N, D]
    :param y: tf.Tensor[B, M, D]
    :return: tensor of shape [B, N, M]
    :rtype tf.Tensor
    >>> from ot.utils import euclidean_distances
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> x = np.random.uniform(-1., 1., [5, 10, 10]).astype(np.float32)
    >>> np.testing.assert_allclose(squared_distances(x, x)[0].numpy(), euclidean_distances(x[0], x[0], squared=True), atol=1e-6)
    """
    # x.shape = [B, N, D]
    xx = tf.reduce_sum(x * x, axis=2, keepdims=True)
    xy = tf.matmul(x, y, transpose_b=True)
    yy = tf.expand_dims(tf.reduce_sum(y * y, axis=-1), 1)
    return tf.clip_by_value(xx - 2 * xy + yy, 0., float('inf'))


@tf.function
def cost(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """ Computes the square distance matrix on the last dimension between two tensors:

    :param x: tf.Tensor[B, N, D]
    :param y: tf.Tensor[B, M, D]
    :return: tensor of shape [B, N, M]
    :rtype tf.Tensor
    """
    return squared_distances(x, y) / 2.
