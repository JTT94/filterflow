import abc

import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.differentiable.optimal_transport.plan import transport


class RegularisedTransform(ResamplerBase, metaclass=abc.ABCMeta):
    """Regularised Transform - docstring to come."""

    # TODO: Document this really nicely
    def __init__(self, epsilon, max_iter=50, convergence_threshold=1e-4, name='RegularisedTransform'):
        """Constructor

        :param epsilon: float
            Regularizer for Sinkhorn iterates
        :param max_iter: int
            max number of iterations in Sinkhorn
        :param convergence_threshold: float
            Fixed point iterates converge when potentials don't move more than this anymore
        """
        self.convergence_threshold = convergence_threshold
        self.max_iter = max_iter
        self.epsilon = epsilon
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

        transported_particles, log_weights = transport(state.particles, state.log_weights, self.epsilon,
                                                       self.convergence_threshold, state.n_particles, self.max_iter)

        return attr.evolve(state, particles=transported_particles, weights=tf.math.exp(log_weights),
                           log_weights=log_weights)
