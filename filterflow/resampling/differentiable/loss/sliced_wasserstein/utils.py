import tensorflow as tf


@tf.function
def random_projections(batch_size, n_projections, dimension):
    """ Generates a batch of random projections vectors on the unit sphere.
    :param batch_size: tf.Tensor
        Number of batches
    :param n_projections: tf.Tensor
        Number of projections to consider
    :param dimension: tf.Tensor
        Dimension of the space
    """
    projections = tf.random.normal([batch_size, n_projections, dimension], 0., 1.)
    projections_norm = tf.math.reduce_euclidean_norm(projections, -1, keepdims=True)
    return projections / projections_norm

@tf.function
def sqeuclidean(x, y):
    return (x - y) ** 2


@tf.function
def norm_1(x, y):
    return tf.abs(x - y)


@tf.function
def emd_1d(w_x, w_y, x, y, metric):
    n_batch_dim = len(x.shape) - 1

    all_values = tf.concat([x, y], -1)
    all_weights = tf.concat([w_x, w_y], -1)
    factor = tf.concat([tf.zeros_like(w_x), tf.ones_like(w_y)], -1)

    all_values_index = tf.argsort(all_values, -1)
    sorted_values = tf.gather(all_values, all_values_index, batch_dims=n_batch_dim)
    sorted_weights = tf.gather(all_weights, all_values_index, batch_dims=n_batch_dim)
    sorted_mask = tf.gather(factor, all_values_index, batch_dims=n_batch_dim)

    cdf_y = tf.math.cumsum(sorted_mask * sorted_weights, -1)[..., :-1]
    cdf_x = tf.math.cumsum((1. - sorted_mask) * sorted_weights, -1)[..., :-1]

    delta = sorted_values[..., 1:] - sorted_values[..., :-1]
    integrand = metric(cdf_y, cdf_x) * delta

    return tf.reduce_sum(integrand, -1)


@tf.function
def sliced_wasserstein(w_x, w_y, x, y, n_projections, metric):
    b, n, d = x.shape
    _, m, _ = y.shape

    projection_vectors = random_projections(b, n_projections, d)

    projected_x = tf.linalg.matmul(projection_vectors, x, transpose_b=True)
    projected_y = tf.linalg.matmul(projection_vectors, y, transpose_b=True)
    tiled_w_x = tf.tile(tf.expand_dims(w_x, 1), [1, n_projections, 1])
    tiled_w_y = tf.tile(tf.expand_dims(w_y, 1), [1, n_projections, 1])

    emd_res = emd_1d(tiled_w_x, tiled_w_y, projected_x, projected_y, metric)
    return tf.reduce_mean(emd_res, -1)
