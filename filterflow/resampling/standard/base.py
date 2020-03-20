import abc
import math

import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase


@tf.function
def discrete_percentile_function(spacings, n_particles, axis, on_log, weights=None, log_weights=None):
    """vectorised resampling function, can be used for systematic/stratified/multinomial resampling
    """
    if on_log:
        cumlogsumexp = tf.math.cumulative_logsumexp(log_weights, axis=axis)
        indices = tf.searchsorted(cumlogsumexp, tf.math.log(spacings))
    else:
        cum_sum = tf.math.cumsum(weights, axis=axis)
        indices = tf.searchsorted(cum_sum, spacings)
    return tf.clip_by_value(indices, 0, n_particles - 1)


@tf.function
def resample_state(state: State, indices: tf.Tensor[int]):
    particles = tf.gather(state.particles, indices, validate_indices=False)
    uniform_weights = tf.fill([state.n_particles, state.batch_size], 1 / state.n_particles)
    uniform_log_weights = tf.fill([state.n_particles, state.batch_size], -math.log(state.n_particles))
    return State(state.n_particles, state.batch_size, state.dimension, particles, uniform_log_weights,
                 uniform_weights, state.log_likelihood, False)


class StandardResamplerBase(ResamplerBase, metaclass=abc.ABCMeta):
    """Abstract ResamplerBase."""

    def __init__(self, on_log=True):
        """Constructor

        :param on_log: bool
            Should the resampling use logweights
        """
        self._on_log = on_log

    @staticmethod
    @abc.abstractmethod
    def _get_spacings(n_particles, batch_size):
        """Spacings variates to give for empirical CDF block selection"""

    def apply(self, state: State):
        """See base class"""
        n_particles = state.n_particles
        batch_size = state.batch_size

        spacings = self._get_spacings(n_particles, batch_size)
        indices = discrete_percentile_function(spacings, n_particles, 0, self._on_log, state.weights, state.log_weights)
        return resample_state(state, indices)
