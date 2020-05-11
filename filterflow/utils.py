import tensorflow as tf


@tf.function
def normalize(weights, axis, log=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    if log:
        return weights - tf.reduce_logsumexp(weights, axis=axis, keepdims=True)
    return weights / tf.reduce_sum(weights, axis=axis)


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
