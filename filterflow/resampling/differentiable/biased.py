import abc

import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase, resample
from filterflow.resampling.differentiable.regularized_transport.plan import transport


class RegularisedTransform(ResamplerBase, metaclass=abc.ABCMeta):
    """Regularised Transform - docstring to come."""

    # TODO: Document this really nicely
    def __init__(self, epsilon, scaling=0.75, max_iter=50, convergence_threshold=1e-4, name='RegularisedTransform'):
        """Constructor

        :param epsilon: float
            Regularizer for Sinkhorn iterates
        :param scaling: float
            Epsilon scaling for sinkhorn iterates
        :param max_iter: int
            max number of iterations in Sinkhorn
        :param convergence_threshold: float
            Fixed point iterates converge when potentials don't move more than this anymore
        """
        self.convergence_threshold = convergence_threshold
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.scaling = scaling
        super(RegularisedTransform, self).__init__(name=name)

    def apply(self, state: State, flags: tf.Tensor):
        """ Resampling method

        :param state State
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """
        # TODO: The real batch_size is the sum of flags. We shouldn't do more operations than we need...
        transport_matrix, _ = transport(state.particles, state.log_weights, self.epsilon, self.scaling,
                                        self.convergence_threshold, state.n_particles, self.max_iter)
        float_n_particles = tf.cast(state.n_particles, float)
        transported_particles = tf.einsum('ijk,ikm->ijm', transport_matrix, state.particles)
        uniform_log_weight = -tf.math.log(float_n_particles) * tf.ones_like(state.log_weights)
        uniform_weights = tf.ones_like(state.weights) / float_n_particles

        resampled_particles, resampled_weights, resampled_log_weights = resample(state.particles,
                                                                                 transported_particles,
                                                                                 state.weights,
                                                                                 uniform_weights,
                                                                                 state.log_weights,
                                                                                 uniform_log_weight,
                                                                                 flags)

        return attr.evolve(state, particles=resampled_particles, weights=resampled_weights,
                           log_weights=resampled_log_weights)
