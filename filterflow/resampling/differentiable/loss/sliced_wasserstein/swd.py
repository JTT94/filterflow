import tensorflow as tf

from filterflow.resampling.differentiable.loss.base import Loss
from filterflow.resampling.differentiable.loss.sliced_wasserstein.utils import random_projections


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

    cdf_y = tf.math.cumsum(sorted_mask * sorted_weights, -1)
    cdf_x = tf.math.cumsum((1. - sorted_mask) * sorted_weights, -1)

    delta = sorted_values[..., 1:] - sorted_values[..., :-1]
    integrand = metric(cdf_y[..., :-1], cdf_x[..., :-1]) * delta
    return tf.reduce_sum(integrand, -1)


@tf.function
def sliced_wasserstein(w_x, w_y, x, y, n_projections, metric):
    b, n, d = x.shape
    _, m, _ = y.shape

    projection_vectors = random_projections(1, n_projections, d)

    projected_x = tf.linalg.matmul(projection_vectors, x, transpose_b=True)
    projected_y = tf.linalg.matmul(projection_vectors, y, transpose_b=True)
    tiled_w_x = tf.tile(tf.expand_dims(w_x, 1), [1, n_projections, 1])
    tiled_w_y = tf.tile(tf.expand_dims(w_y, 1), [1, n_projections, 1])

    emd_res = emd_1d(tiled_w_x, tiled_w_y, projected_x, projected_y, metric)
    return tf.reduce_sum(emd_res, -1)


class SlicedWassersteinDistance(Loss):
    def __init__(self, n_slices, metric, name='SlicedWassersteinDistance'):
        super(SlicedWassersteinDistance, self).__init__(name=name)
        self._n_slices = tf.cast(n_slices, tf.dtypes.int32)
        self._metric = metric

    def __call__(self, _log_w_x, w_x, x, _log_w_y, w_y, y):
        return sliced_wasserstein(w_x, w_y, x, y, self._n_slices, self._metric)
