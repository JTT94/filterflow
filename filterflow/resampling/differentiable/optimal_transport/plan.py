import tensorflow as tf

from filterflow.resampling.differentiable.optimal_transport.sinkhorn import sinkhorn_potentials
from filterflow.resampling.differentiable.optimal_transport.utils import cost


@tf.function
def fillna(tensor, value):
    return tf.where(tf.math.is_nan(tensor), value, tensor)


@tf.function
def transport_from_potentials(x, f, g, eps, logw, n):
    """
    To get the transported particles from the sinkhorn iterates
    :param x: tf.Tensor[B, N, D]
        Input: the state variable
    :param f: tf.Tensor[B, N]
        Potential, output of the sinkhorn iterates
    :param g: tf.Tensor[B, N]
        Potential, output of the sinkhorn iterates
    :param eps: float
    :param logw: torch.Tensor[N]
    :param n: int
    :return: tf.Tensor[N, D], tf.Tensor[N]
    """
    float_n = tf.cast(n, float)

    cost_matrix = cost(x, x)
    fg = tf.einsum('ij,ik->ijk', f, g)
    temp = (fg - cost_matrix) / eps
    # Total contribution can't be less than 1/n^3
    transport_matrix = tf.math.exp(temp + tf.expand_dims(logw, 2))
    transport_matrix = fillna(transport_matrix / tf.reduce_sum(transport_matrix, axis=2, keepdims=True),
                              tf.constant(0.))
    res = tf.einsum('ijk,ikm->ijm', transport_matrix, x)

    uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(logw)
    return res, uniform_log_weight


@tf.function
def solve_for_state(x, logw, eps, threshold, max_iter, n):
    """
    :param x: tf.Tensor[N, D]
        The input
    :param logw: tf.Tensor[N]
        The degenerate logweights
    :param eps: float
    :param threshold: float
    :param max_iter: int
    :param n: int
    :return: torch.Tensor[N], torch.Tensor[N]
        the potentials
    """
    float_n = tf.cast(n, float)
    uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(logw)
    return sinkhorn_potentials(uniform_log_weight, x, logw, x, eps, threshold, max_iter)


@tf.function
def transport(x, logw, eps, threshold, n, max_iter):
    """
    Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
    :param x: tf.Tensor[N, D]
        The input
    :param logw: tf.Tensor[N]
        The degenerate logweights
    :param eps: float
    :param threshold: float
    :param n: int
    :param max_iter: int
    """
    alpha, beta = solve_for_state(x, logw, eps, threshold, max_iter, n)
    return transport_from_potentials(x, alpha, beta, eps, logw, n)
