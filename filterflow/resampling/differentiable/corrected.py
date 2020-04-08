import abc

import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.differentiable.biased import apply_transport_matrix
from filterflow.resampling.differentiable.regularized_transport.plan import transport


class CorrectedRegularizedTransform(ResamplerBase, metaclass=abc.ABCMeta):
    """Regularised Transform - docstring to come."""

    # TODO: Document this really nicely
    def __init__(self, epsilon, scaling, max_iter, convergence_threshold, ricatti_solver, name='RegularisedTransform'):
        """Constructor

        :param epsilon: float
            Regularizer for Sinkhorn iterates
        :param scaling: float
            Epsilon scaling for sinkhorn iterates
        :param max_iter: int
            max number of iterations in Sinkhorn
        :param convergence_threshold: float
            Fixed point iterates converge when potentials don't move more than this anymore
        :param ricatti_solver: filterflow.resampling.differentiable.ricatti.solver.RicattiSolver
        """
        self.convergence_threshold = tf.cast(convergence_threshold, float)
        self.max_iter = tf.cast(max_iter, tf.dtypes.int32)
        self.epsilon = tf.cast(epsilon, float)
        self.scaling = tf.cast(scaling, float)
        self.ricatti_solver = ricatti_solver
        super(CorrectedRegularizedTransform, self).__init__(name=name)

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
                                        self.convergence_threshold, self.max_iter, state.n_particles)

        transport_correction = self.ricatti_solver(transport_matrix, state.weights)
        return apply_transport_matrix(state, transport_matrix + transport_correction, flags)
