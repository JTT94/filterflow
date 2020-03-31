import tensorflow as tf


@tf.function
def _diff_1(a):
    return a[:, 1:] - a[:, :-1]


@tf.function
def random_projections(batch_size, n_projections, embedding_dimension):
    """ Generates a batch of random projections vectors on the unit sphere.
    :param batch_size: tf.Tensor
        Number of batches
    :param n_projections: tf.Tensor
        Number of projections to consider
    :param embedding_dimension: tf.Tensor
        Dimension of the embedding space space
    """
    projections = tf.random.normal([batch_size, n_projections, embedding_dimension], 0., 1.)
    projections_norm = tf.math.reduce_euclidean_norm(projections, 1, keepdims=True)
    return projections / projections_norm


def _cdf_distance(u_values, v_values, u_weights, v_weights):
    """Wildly adapted from Scipy.
    """
    batch_size = u_values.shape[0]
    n = u_values.shape[1]

    u_sorter = tf.argsort(u_values, axis=1)
    v_sorter = tf.argsort(v_values, axis=1)

    indices = tf.expand_dims(tf.range(0, batch_size, 1, dtype=u_sorter.dtype), 1)
    indices = tf.tile(indices, [1, n])

    u_sorter = tf.stack([indices, u_sorter], -1)
    v_sorter = tf.stack([indices, v_sorter], -1)

    all_values = tf.concat((u_values, v_values), 1)
    all_values = tf.sort(all_values, 1)

    # Compute the differences between pairs of successive values of u and v.
    deltas = _diff_1(all_values)

    u_to_search = tf.gather_nd(u_values, u_sorter)
    v_to_search = tf.gather_nd(v_values, v_sorter)

    u_cdf_indices = tf.searchsorted(u_to_search, all_values[:, :-1], 'right')
    v_cdf_indices = tf.searchsorted(v_to_search, all_values[:, :-1], 'right')

    cdf_indices = tf.expand_dims(tf.range(0, batch_size, 1, dtype=u_sorter.dtype), 1)
    cdf_indices = tf.tile(cdf_indices, [1, 2 * n - 1])

    u_cdf_indices = tf.stack([cdf_indices, u_cdf_indices], -1)
    v_cdf_indices = tf.stack([cdf_indices, v_cdf_indices], -1)
    # Calculate the CDFs of u and v using their weights, if specified.
    zeros = tf.zeros([batch_size, 1], dtype=u_weights.dtype)

    u_sorted_cumweights = tf.concat([zeros, tf.cumsum(tf.gather_nd(u_weights, u_sorter))], 1)

    u_cdf = tf.gather_nd(u_sorted_cumweights, u_cdf_indices) / tf.gather(u_sorted_cumweights, [n], axis=1)

    v_sorted_cumweights = tf.concat([zeros, tf.cumsum(tf.gather_nd(v_weights, v_sorter))], 1)
    v_cdf = tf.gather_nd(v_sorted_cumweights, v_cdf_indices) / tf.gather(v_sorted_cumweights, [n], axis=1)

    return tf.sqrt(tf.reduce_sum(tf.square(u_cdf - v_cdf) * deltas, 1))



@tf.function
def emd_1d(x, y, w_x, w_y):
    """ Computes the EMD between two weighted samples. This is highly inspired by the scipy implementation
    :param x: tf.Tensor
        first component
    :param y: tf.Tensor
        second component
    :param w_x: tf.Tensor
        Weights for first component
    :param w_y: tf.Tensor
        Weights for second component
    """
