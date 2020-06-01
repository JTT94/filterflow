import tensorflow as tf

from filterflow.constants import MIN_RELATIVE_LOG_WEIGHT, MIN_ABSOLUTE_LOG_WEIGHT, MIN_RELATIVE_WEIGHT, \
    MIN_ABSOLUTE_WEIGHT


@tf.function
def _normalize(weights, axis, log=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    if log:
        normalizer = tf.reduce_logsumexp(weights, axis=axis, keepdims=True)
        return weights - normalizer
    normalizer = tf.reduce_sum(weights, axis=axis)
    return weights / normalizer


@tf.function
def normalize(weights, axis, n, log=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    float_n = tf.cast(n, float)

    if log:
        normalized_weights = tf.clip_by_value(_normalize(weights, axis, True), tf.constant(-1e3), tf.constant(0.))
        stop_gradient_mask = normalized_weights < tf.maximum(MIN_ABSOLUTE_LOG_WEIGHT, MIN_RELATIVE_LOG_WEIGHT * float_n)
    else:
        normalized_weights = _normalize(weights, axis, False)
        stop_gradient_mask = normalized_weights < tf.maximum(MIN_ABSOLUTE_WEIGHT, MIN_RELATIVE_WEIGHT ** float_n)
    float_stop_gradient_mask = tf.cast(stop_gradient_mask, float)
    return tf.stop_gradient(float_stop_gradient_mask * normalized_weights) + (
                1. - float_stop_gradient_mask) * normalized_weights


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
    var = tf.reduce_sum(tf.expand_dims(state.weights, -1) * centered_state ** 2, axis=-2, keepdims=keepdims)
    return tf.sqrt(var)
