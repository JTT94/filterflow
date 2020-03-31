import tensorflow as tf


@tf.function
def diameter(tensor):
    stddevs = tf.math.reduce_std(tensor, 1)
    return tf.reduce_max(stddevs, 1)

@tf.function
def dampening(ε, ρ):
    return 1 / ( 1 + ε / ρ )

@tf.function
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
    temp_val = f_ - cost_matrix / tf.reshape(epsilon, (b, 1, 1))
    log_sum_exp = tf.reduce_logsumexp(temp_val, axis=2)
    return -tf.reshape(epsilon, (b, 1)) * log_sum_exp


@tf.function
def squared_distances(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """ Computes the square distance matrix on the last dimension between two tensors:

    :param x: tf.Tensor[B, N, D]
    :param y: tf.Tensor[B, M, D]
    :return: tensor of shape [B, N, M]
    :rtype tf.Tensor
    """
    # x.shape = [B, N, D]
    xx = tf.reduce_sum(x * x, axis=2, keepdims=True)
    xy = tf.einsum('bnd,bmd->bnm', x, y)
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
    return squared_distances(x, y) / tf.constant(2.)
