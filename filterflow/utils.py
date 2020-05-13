import tensorflow as tf

from filterflow.constants import MIN_RELATIVE_LOG_WEIGHT, MIN_ABSOLUTE_LOG_WEIGHT, MIN_RELATIVE_WEIGHT, \
    MIN_ABSOLUTE_WEIGHT


@tf.function
def normalize(weights, axis, n, log=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    float_n = tf.cast(n, float)
    if log:
        clip_min = tf.maximum(MIN_ABSOLUTE_LOG_WEIGHT, tf.math.log(float_n) * MIN_RELATIVE_LOG_WEIGHT)
        clipped_weights = tf.maximum(weights, clip_min)
        normalizer = tf.reduce_logsumexp(clipped_weights, axis=axis, keepdims=True)
        return clipped_weights - normalizer
    clip_min = tf.maximum(MIN_ABSOLUTE_WEIGHT, float_n * MIN_RELATIVE_WEIGHT)
    clipped_weights = tf.maximum(clip_min, weights)
    normalizer = tf.reduce_sum(clipped_weights, axis=axis)
    return clipped_weights / normalizer


@tf.function
def mean(state, keepdims=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    return tf.reduce_sum(tf.expand_dims(state.weights, -1) * state.particles, axis=-2, keepdims=keepdims)


@tf.function
def std(state, avg=None, keepdims=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    if avg is None:
        avg = mean(state, keepdims=True)
    centered_state = state.particles - avg
    var = tf.reduce_sum(tf.expand_dims(state.weights, -1) * centered_state**2, axis=-2, keepdims=keepdims)
    return tf.sqrt(var)
