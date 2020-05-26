import abc

import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase, resample


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


class StandardResamplerBase(ResamplerBase, metaclass=abc.ABCMeta):
    """Abstract ResamplerBase."""
    DIFFERENTIABLE = False

    def __init__(self, name, on_log):
        """Constructor

        :param on_log: bool
            Should the resampling use log weights
        :param stop_gradient: bool
            Should the resampling step propagate the stitched gradients or not
       """
        self._on_log = on_log
        super(StandardResamplerBase, self).__init__(name=name)

    @staticmethod
    @abc.abstractmethod
    def _get_spacings(n_particles, batch_size, seed):
        """Spacings variates to give for empirical CDF block selection"""

    def apply(self, state: State, flags: tf.Tensor, seed=None):
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

        spacings = self._get_spacings(n_particles, batch_size, seed)
        # TODO: We should be able to get log spacings directly to always stay in log space.
        indices = _discrete_percentile_function(spacings, n_particles, self._on_log, state.weights,
                                                state.log_weights)

        ancestor_indices = tf.where(tf.reshape(flags, [-1, 1]), indices, tf.reshape(tf.range(n_particles), [1, -1]))

        new_particles = tf.gather(state.particles, indices, axis=1, batch_dims=1, validate_indices=False)

        float_n_particles = tf.cast(n_particles, float)
        uniform_weights = tf.ones_like(state.weights) / float_n_particles
        uniform_log_weights = tf.zeros_like(state.log_weights) - tf.math.log(float_n_particles)

        resampled_particles = resample(state.particles, new_particles, flags)
        resampled_weights = resample(state.weights, uniform_weights, flags)
        resampled_log_weights = resample(state.log_weights, uniform_log_weights, flags)

        additional_variables = {}

        for additional_state_variable in state.ADDITIONAL_STATE_VARIABLES:
            state_variable = getattr(state, additional_state_variable)
            new_state_variable = tf.gather(state_variable, indices, axis=1, batch_dims=1, validate_indices=False)
            additional_variables[additional_state_variable] = resample(state_variable, new_state_variable, flags)

        return attr.evolve(state, particles=resampled_particles, weights=resampled_weights,
                           log_weights=resampled_log_weights, ancestor_indices=ancestor_indices, **additional_variables)
