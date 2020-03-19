import abc

import tensorflow as tf


@tf.function
def _weighted_avg(values, weights, axis, assume_normalized=True):
    res = tf.reduce_sum(values * weights, axis, keepdims=True)
    if assume_normalized:
        return res
    else:
        return res / tf.reduce_sum(weights, axis)


@tf.function
def _weighted_covariance(values, weights, axis, assume_normalized=True):
    weighted_avg = _weighted_avg(values, weights, axis, assume_normalized=True)
    res = tf.reduce_sum((values - weighted_avg) ** 2 * weights, axis, keepdims=True)
    if assume_normalized:
        return res
    else:
        return res / tf.reduce_sum(weights, axis)


class State(object):
    """Particle Filter State
    State encapsulates the information about the particle filter current state.
    """
    __slots__ = ['_n_particles', '_batch_size', '_dimension', '_particles', '_log_weights', '_weights',
                 '_log_likelihood']

    def __init__(self, n_particles: int, batch_size: int, dimension: int, particles: tf.Tensor,
                 log_weights: tf.Tensor = None, weights: tf.Tensor = None, log_likelihood: tf.Tensor = None,
                 check_shapes=False):
        self._n_particles = n_particles
        self._batch_size = batch_size
        self._dimension = dimension
        self._particles = particles
        self._log_weights = log_weights or tf.math.log(weights)
        self._weights = weights or tf.math.exp(log_weights)
        self._log_likelihood = log_likelihood or tf.constant(0.)
        if check_shapes:
            self._check_shapes()

    def _check_shapes(self):
        pass

    def mean(self):
        """Weighted Average of the State

        :return:
            Weighted Average
        :rtype
            tf.Tensor
        """
        return _weighted_avg(self._particles, self._weights, 0, assume_normalized=True)

    def covariance(self):
        """Weighted Covariance of the State

        :return
            Weighted Average
        :rtype
            tf.Tensor
        """
        return _weighted_covariance(self._particles, self._weights, 0, assume_normalized=True)

    @property
    def shape(self):
        return [self._batch_size, self._n_particles, self._dimension]

    @property
    def particles(self):
        return self._particles

    @property
    def log_weights(self):
        return self._log_weights

    @property
    def log_likelihood(self):
        return self._log_likelihood

    @property
    def weights(self):
        return self._weights

    @property
    def n_particles(self):
        return self._n_particles

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dimension(self):
        return self._dimension


class ObservationBase(metaclass=abc.ABCMeta):
    __slots__ = ['_observations', '_dimension']

    def __init__(self, observations, dimension):
        self._observations = observations
        self._dimension = dimension

    @property
    def shape(self):
        return [1, self._dimension]

    @property
    def observations(self):
        return tf.reshape(self._observations, (1, self._dimension))


class InputsBase(metaclass=abc.ABCMeta):
    """Container interface to implement an arbitrary collection of inputs to give to the filter at time of prediction
    """
    pass

