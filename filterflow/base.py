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
                 '_log_likelihoods', '_check_shapes']

    def __init__(self, batch_size: int, n_particles: int, dimension: int, particles: tf.Tensor,
                 log_weights: tf.Tensor = None, weights: tf.Tensor = None, log_likelihoods: tf.Tensor = None,
                 check_shapes: bool = True):
        self._batch_size = batch_size
        self._n_particles = n_particles
        self._dimension = dimension
        self._particles = particles
        self._log_weights = log_weights if log_weights is not None else tf.math.log(weights)
        self._weights = weights if weights is not None else tf.math.exp(log_weights)
        self._log_likelihoods = log_likelihoods if log_likelihoods is not None else tf.zeros(batch_size)
        self._check_shapes = check_shapes
        if check_shapes:
            self._do_check_shapes()

    def _do_check_shapes(self):
        weights_shape = self._weights.shape.as_list()
        log_weights_shape = self._log_weights.shape.as_list()
        shape = self.shape
        assert shape == [self._batch_size, self._n_particles, self._dimension]
        assert weights_shape == log_weights_shape
        assert shape[0] == weights_shape[0]
        assert shape[1] == weights_shape[1]
        assert self._log_likelihoods.shape.as_list()[0] == self._batch_size

    def mean(self):
        """Weighted Average of the State

        :return:
            Weighted Average
        :rtype
            tf.Tensor
        """
        return _weighted_avg(self._particles, self._weights, 1, assume_normalized=True)

    def covariance(self):
        """Weighted Covariance of the State

        :return
            Weighted Average
        :rtype
            tf.Tensor
        """
        return _weighted_covariance(self._particles, self._weights, 1, assume_normalized=True)

    @property
    def shape(self):
        return self._particles.shape.as_list()

    @property
    def particles(self):
        return self._particles

    @property
    def log_weights(self):
        return self._log_weights

    @property
    def log_likelihoods(self):
        return self._log_likelihoods

    @property
    def weights(self):
        return self._weights

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_particles(self):
        return self._n_particles

    @property
    def dimension(self):
        return self._dimension

    @property
    def check_shapes(self):
        return self._check_shapes


class ObservationBase(metaclass=abc.ABCMeta):
    __slots__ = ['_observation', '_dimension']

    def __init__(self, observation, dimension):
        self._observation = observation
        self._dimension = dimension

    @property
    def dimension(self):
        return self._dimension

    @property
    def observation(self):
        return self._observation


class InputsBase(metaclass=abc.ABCMeta):
    """Container interface to implement an arbitrary collection of inputs to give to the filter at time of prediction
    """
    pass
