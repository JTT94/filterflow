import abc
import math
from typing import List

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


# @tf.function
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

    def __init__(self, n_particles, on_log=True):
        """Constructor

        :param n_particles: int
            Number of particles in states
        :param on_log: bool
            Should the resampling use log weights
        """
        self._n_particles = n_particles
        self._on_log = on_log

    @staticmethod
    @abc.abstractmethod
    def _get_spacings(n_particles, batch_size):
        """Spacings variates to give for empirical CDF block selection"""

    def apply(self, states: List[State], flags: tf.Tensor):
        """ Resampling method

        :param states: List[State]
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: list of resampled states
        :rtype: List[State]
        """
        batch_size = len(states)
        # TODO: The real batch_size is the sum of flags. We shouldn't do more operations than we need...

        spacings = self._get_spacings(self._n_particles, batch_size)
        stacked_weights = tf.stack([state.weights for state in states], 0)
        stacked_log_weights = tf.stack([state.log_weights for state in states], 0)
        stacked_particles = tf.stack([state.particles for state in states], 0)
        indices = _discrete_percentile_function(spacings, self._n_particles, self._on_log, stacked_weights,
                                                stacked_log_weights)
        resampled_particles, resampled_weights, resampled_log_weights = _resample(stacked_particles,
                                                                                  stacked_weights,
                                                                                  stacked_log_weights,
                                                                                  indices,
                                                                                  flags,
                                                                                  self._n_particles,
                                                                                  batch_size)
        particles = tf.unstack(resampled_particles, axis=0)
        weights = tf.unstack(resampled_weights, axis= 0)
        log_weights = tf.unstack(resampled_log_weights, axis= 0)
        states = [State(state.dimension, state_particles, state_log_weights, state_weights, state.log_likelihood, False)
                  for state, state_particles, state_log_weights, state_weights in
                  zip(states, particles, log_weights, weights)]

        return states
