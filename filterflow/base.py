"""
This file is a technical script to create data structures compatible with tf.function.
"""
import abc
from functools import partial

import attr
import tensorflow as tf


def _pairwise(iterable):
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)


class Module(tf.Module, metaclass=abc.ABCMeta):
    """
    Base class for filterflow Modules - __call__ is ignored
    """

    def __call__(self, *args, **kwargs):
        pass


@attr.s(frozen=True)
class DataSeries(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def write(self, t, data_point):
        """Inferface method"""

    @abc.abstractmethod
    def read(self, t):
        """Inferface method"""

    @abc.abstractmethod
    def stack(self):
        """Inferface method"""


def _non_scalar_validator(_instance, attribute, value, ndim=None):
    if not hasattr(value, 'shape'):
        raise ValueError(f"value {value} for attribute {attribute} should have an attribute shape.")
    value_shape = value.shape
    if ndim is not None:
        assert len(value_shape) == ndim, f"value {value} for attribute {attribute} should have ndim {ndim}."


_dim_3_validator = partial(_non_scalar_validator, ndim=3)
_dim_2_validator = partial(_non_scalar_validator, ndim=2)
_dim_1_validator = partial(_non_scalar_validator, ndim=1)


@attr.s(frozen=True)
class State:
    """Particle Filter State
    State encapsulates the information about the particle filter current state.
    """

    particles = attr.ib(validator=_dim_3_validator)
    log_weights = attr.ib(validator=_dim_2_validator)
    weights = attr.ib(validator=_dim_2_validator)
    log_likelihoods = attr.ib(validator=_dim_1_validator)

    _batch_size = attr.ib(default=None)
    _n_particles = attr.ib(default=None)
    _dimension = attr.ib(default=None)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_particles(self):
        return self._n_particles

    @property
    def dimension(self):
        return self._dimension

    def __attrs_post_init__(self):
        particles_batch, particles_n_particles, particles_dimension = self.particles.shape

        if self._batch_size is None:
            object.__setattr__(self, '_batch_size', particles_batch)

        if self._n_particles is None:
            object.__setattr__(self, '_n_particles', particles_n_particles)

        if self._dimension is None:
            object.__setattr__(self, '_dimension', particles_dimension)


@attr.s(frozen=True)
class StateSeries(DataSeries, metaclass=abc.ABCMeta):
    batch_size = attr.ib()
    n_particles = attr.ib()
    dimension = attr.ib()

    _particles = attr.ib(default=None)
    _log_weights = attr.ib(default=None)
    _weights = attr.ib(default=None)
    _log_likelihoods = attr.ib(default=None)

    DTYPE = None

    @property
    def particles(self):
        return self._particles

    @property
    def log_weights(self):
        return self._log_weights

    @property
    def weights(self):
        return self._weights

    @property
    def log_likelihoods(self):
        return self._log_likelihoods

    def _set_ta(self, attr, shape):
        ta = tf.TensorArray(self.DTYPE,
                            size=0,
                            dynamic_size=True, clear_after_read=False, element_shape=tf.TensorShape(shape))
        object.__setattr__(self, attr, ta)

    def __attrs_post_init__(self):
        # particles
        particles_shape = [self.batch_size, self.n_particles, self.dimension]
        if self._particles is None:
            self._set_ta('_particles', particles_shape)

        # weights
        weights_shape = [self.batch_size, self.n_particles]
        if self._weights is None:
            self._set_ta('_weights', weights_shape)

        # log_weights
        log_weights_shape = [self.batch_size, self.n_particles]
        if self._log_weights is None:
            self._set_ta('_log_weights', log_weights_shape)

        # log_likelihoods
        log_lik_shape = [self.batch_size]
        if self._log_likelihoods is None:
            self._set_ta('_log_likelihoods', log_lik_shape)

    def write(self, t, state):
        particles = self._particles.write(t, state.particles)
        log_weights = self._log_weights.write(t, state.log_weights)
        weights = self._weights.write(t, state.weights)
        log_likelihoods = self._log_likelihoods.write(t, state.log_likelihoods)
        return self.__class__(self.batch_size, self.n_particles, self.dimension, particles, log_weights, weights,
                              log_likelihoods)

    def stack(self):
        particles = self._particles.stack()
        log_weights = self._log_weights.stack()
        weights = self._weights.stack()
        log_likelihoods = self._log_likelihoods.stack()
        return self.__class__(self.batch_size, self.n_particles, self.dimension, particles, log_weights, weights,
                              log_likelihoods)

    def read(self, t):
        if isinstance(self._particles, tf.TensorArray):
            particles = self._particles.read(t)
            log_weights = self._log_weights.read(t)
            weights = self._weights.read(t)
            log_likelihoods = self._log_likelihoods.read(t)
        else:
            particles = self._particles[t]
            log_weights = self._log_weights[t]
            weights = self._weights[t]
            log_likelihoods = self._log_likelihoods[t]

        state = State(particles=particles,
                      log_weights=log_weights,
                      weights=weights,
                      log_likelihoods=log_likelihoods)
        return state


@attr.s(frozen=True)
class FloatStateSeries(StateSeries):
    DTYPE = tf.dtypes.float32


@attr.s(frozen=True)
class DoubleStateSeries(StateSeries):
    DTYPE = tf.dtypes.float64


@attr.s(frozen=True)
class Observation:
    observation = attr.ib(validator=_non_scalar_validator)
    _shape = attr.ib(default=None)

    @property
    def shape(self):
        return self._shape

    def __attrs_post_init__(self):

        observation = self.observation

        if not hasattr(observation, 'shape'):
            raise ValueError(f"value {observation} for attribute observation should have an attribute shape. "
                             f"If unsure use convert=True")

        if self._shape is None:
            object.__setattr__(self, '_shape', self.observation.shape)
        object.__setattr__(self, '_observation', observation)


@attr.s(frozen=True)
class ObservationSeries(DataSeries, metaclass=abc.ABCMeta):
    DTYPE = None

    shape = attr.ib()
    _observation = attr.ib(default=None)

    @property
    def observation(self):
        return self._observation

    def __attrs_post_init__(self):
        if self._observation is None:
            ta = tf.TensorArray(self.DTYPE,
                                size=0,
                                dynamic_size=True,
                                clear_after_read=False,
                                tensor_array_name='ObservationSeries',
                                element_shape=tf.TensorShape(self.shape))
            object.__setattr__(self, '_observation', ta)

    def write(self, t, observation):
        observation = self._observation.write(t, observation.observation)
        return self.__class__(self.shape, observation)

    def stack(self):
        return self.__class__(self.shape, self._observation.stack())

    def read(self, t):
        observation = self._observation[t]
        return Observation(observation)


@attr.s(frozen=True)
class FloatObservationSeries(ObservationSeries):
    DTYPE = tf.dtypes.float32


@attr.s(frozen=True)
class DoubleObservationSeries(ObservationSeries):
    DTYPE = tf.dtypes.float64


# TODO : Have a think about what inputs we want exactly - e.g. use a dynamic classifier example to see what we need
@attr.s
class InputsBase(metaclass=abc.ABCMeta):
    """Container interface to implement an arbitrary collection of inputs to give to the filter at time of prediction
    """
