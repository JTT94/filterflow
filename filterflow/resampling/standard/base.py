import abc
import math

import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase


@tf.function
def _discrete_percentile_function(spacings, n_particles, on_log, weights=None, log_weights=None):
    """vectorised resampling function, can be used for systematic/stratified/multinomial resampling
    """
    if on_log:
        cumlogsumexp = tf.math.cumulative_logsumexp(log_weights, axis=1)
        log_spacings = tf.math.log(spacings)
        indices = tf.searchsorted(cumlogsumexp, log_spacings, side='left')
    else:
        cum_sum = tf.math.cumsum(weights, axis=1)
        indices = tf.searchsorted(cum_sum, spacings, side='left')
    return tf.clip_by_value(indices, 0, n_particles - 1)


@tf.function
def _resample(particles: tf.Tensor, weights: tf.Tensor, log_weights: tf.Tensor, indices: tf.Tensor,
              flags: tf.Tensor, n_particles: int, batch_size: int):
    uniform_weights = tf.ones_like(weights) / n_particles
    uniform_log_weights = tf.zeros_like(log_weights) - math.log(n_particles)
    resampled_particles = tf.gather(particles, indices, axis=1, batch_dims=1, validate_indices=False)

    particles = tf.where(tf.reshape(flags, [-1, 1, 1]),
                         resampled_particles,
                         particles)

    weights = tf.where(tf.reshape(flags, [-1, 1]),
                       uniform_weights,
                       weights)

    log_weights = tf.where(tf.reshape(flags, [-1, 1]),
                           uniform_log_weights,
                           log_weights)

    return particles, weights, log_weights


class StandardResamplerBase(ResamplerBase, metaclass=abc.ABCMeta):
    """Abstract ResamplerBase."""

    def __init__(self, on_log=True):
        """Constructor

        :param on_log: bool
            Should the resampling use log weights
        """
        self._on_log = on_log

    @staticmethod
    @abc.abstractmethod
    def _get_spacings(n_particles, batch_size):
        """Spacings variates to give for empirical CDF block selection"""

    def apply(self, state: State, flags: tf.Tensor):
        """ Resampling method

        :param state State
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """
        batch_size = state.batch_size
        n_particles = state.n_particles
        # TODO: The real batch_size is the sum of flags. We shouldn't do more operations than we need...

        spacings = self._get_spacings(n_particles, batch_size)
        # TODO: We should be able to get log spacings directly to always stay in log space.
        indices = _discrete_percentile_function(spacings, n_particles, self._on_log, state.weights,
                                                state.log_weights)
        resampled_particles, resampled_weights, resampled_log_weights = _resample(state.particles,
                                                                                  state.weights,
                                                                                  state.log_weights,
                                                                                  indices,
                                                                                  flags,
                                                                                  n_particles,
                                                                                  batch_size)

        return State(batch_size, n_particles, state.dimension, resampled_particles, resampled_log_weights,
                     resampled_weights, state.log_likelihoods, check_shapes=state.check_shapes)
