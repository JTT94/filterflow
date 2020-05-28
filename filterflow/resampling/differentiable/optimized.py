import abc

import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase, resample
from filterflow.resampling.differentiable.optimizer.base import OptimizerBase


class OptimizedPointCloud(ResamplerBase, metaclass=abc.ABCMeta):
    """Optimized Point Cloud - docstring to come."""

    DIFFERENTIABLE = True

    # TODO: Document this really nicely
    def __init__(self, optimizer: OptimizerBase, intermediate_resampler: ResamplerBase, name='OptimizedPointCloud'):
        """Constructor

        :param optimizer: OptimizerBase
            a tf.Module that takes (log_w_x, w_x, x, log_w_y, w_y, y) and optimizes a loss w.r.t. x
        :param intermediate_resampler: ResamplerBase
            Provides the initial point cloud to optimize
        """
        self.optimizer = optimizer
        self.intermediate_resampler = intermediate_resampler
        super(OptimizedPointCloud, self).__init__(name=name)

    def apply(self, state: State, flags: tf.Tensor, seed=None):
        """ Resampling method

        :param state State
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """
        intermediate_state = self.intermediate_resampler.apply(state, flags)

        optimized_particles = self.optimizer(intermediate_state.log_weights, intermediate_state.weights,
                                             intermediate_state.particles, state.log_weights, state.weights,
                                             state.particles)
        float_n_particles = tf.cast(state.n_particles, float)
        uniform_log_weights = -tf.math.log(float_n_particles) * tf.ones_like(state.log_weights)
        uniform_weights = tf.ones_like(state.weights) / float_n_particles

        resampled_particles = resample(intermediate_state.particles, optimized_particles, flags)
        resampled_weights = resample(intermediate_state.weights, uniform_weights, flags)
        resampled_log_weights = resample(intermediate_state.log_weights, uniform_log_weights, flags)

        return attr.evolve(state, particles=resampled_particles, weights=resampled_weights,
                           log_weights=resampled_log_weights)
