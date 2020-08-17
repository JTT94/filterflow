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
def _native_mean(weights, particles, keepdims):
    weights = tf.expand_dims(weights, -1)
    res = tf.reduce_sum(weights * particles, axis=-2, keepdims=keepdims)
    return res


@tf.function
def _log_mean(log_weights, particles, keepdims):
    max_log_weights = tf.stop_gradient(tf.reduce_max(log_weights, axis=-1, keepdims=True))
    weights = tf.exp(log_weights - max_log_weights)
    if keepdims:
        max_log_weights = tf.expand_dims(max_log_weights, -1)

    temp = particles * tf.expand_dims(weights, -1)
    temp = tf.reduce_sum(temp, -2, keepdims=keepdims)
    res = max_log_weights + tf.math.log(temp)

    return res


@tf.function
def mean(state, keepdims=True, is_log=False):
    """Returns the weighted averaged of the state"""
    return mean_raw(state.particles, state.log_weights, keepdims, is_log)


@tf.function
def mean_raw(particles, log_weights, keepdims=True, is_log=False):
    """Returns the weighted averaged of the state"""
    if is_log:
        return _log_mean(log_weights, particles, keepdims)
    else:
        return _native_mean(tf.exp(log_weights), particles, keepdims)


@tf.function
def std_raw(particles, log_weights, avg=None, keepdims=True, is_log=False):
    """Normalises weights, either expressed in log terms or in their natural space"""
    if avg is None:
        avg = mean_raw(particles, log_weights, keepdims=True, is_log=False)
    centered_state = particles - avg
    if is_log:
        var = _log_mean(log_weights, centered_state ** 2, keepdims=keepdims)
        return 0.5 * var
    else:
        var = _native_mean(tf.exp(log_weights), centered_state ** 2, keepdims=keepdims)
        return tf.sqrt(var)


@tf.function
def std(state, avg=None, keepdims=True, is_log=False):
    """Normalises weights, either expressed in log terms or in their natural space"""
    return std_raw(state.particles, state.log_weights, avg, keepdims, is_log)
