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
    :
    :return: the transportation matrix
    :rtype: tf.Tensor[B, N, N]

    """

    cost_matrix = cost(x, x)
    fg = (tf.expand_dims(f, 2) + tf.expand_dims(g, 1))  # fg = f + g.T
    temp = (fg - cost_matrix) / eps
    temp = temp - tf.reduce_logsumexp(temp, 1, keepdims=True) + tf.math.log(n)
    # We "divide the transport matrix by its col-wise sum to make sure that weights normalise to logw.
    transport_matrix = tf.math.exp(temp + tf.expand_dims(logw, 1))
    return transport_matrix


@tf.function
def transport(x, logw, eps, scaling, threshold, max_iter, n):
    """
    Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
    :param x: tf.Tensor[B, N, D]
        The input
    :param logw: tf.Tensor[B, N]
        The degenerate logweights
    :param eps: float
    :param scaling: float
    :param threshold: float
    :param n: int
    :param max_iter: int

    :return transport matrix
    :rtype tf.Tensor[B, N, N]
    """
    float_n = tf.cast(n, float)
    uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(logw)

    alpha, beta, total_iterations = sinkhorn_potentials(logw, x, uniform_log_weight, x, eps, scaling, threshold,
                                                        max_iter)
    transport_matrix = transport_from_potentials(x, alpha, beta, eps, logw, float_n)
    return transport_matrix, total_iterations
