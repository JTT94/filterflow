import tensorflow as tf


@tf.function
def normalize(weights, axis, log=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    if log:
        return weights - tf.reduce_logsumexp(weights, axis=axis, keepdims=True)
    return weights / tf.reduce_sum(weights, axis=axis)
