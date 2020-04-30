import abc

import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.differentiable.biased import apply_transport_matrix
from filterflow.resampling.differentiable.regularized_transport.plan import transport
from filterflow.resampling.differentiable.ricatti.solver import PetkovSolver


class CorrectedRegularizedTransform(ResamplerBase, metaclass=abc.ABCMeta):
    """Corrected Regularised (covariance) Transform - docstring to come."""

    DIFFERENTIABLE = True

    # TODO: Document this really nicely
    def __init__(self, epsilon, scaling=0.75, max_iter=100, convergence_threshold=1e-3, ricatti_solver=None,
                 propagate_correction_gradient=True, name='RegularisedTransform'):
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
        :param propagate_correction_gradient: should you propagate the correction factor gradient
        """
        self.convergence_threshold = tf.cast(convergence_threshold, float)
        self.max_iter = tf.cast(max_iter, tf.dtypes.int32)
        self.epsilon = tf.cast(epsilon, float)
        self.scaling = tf.cast(scaling, float)
        self.propagate_correction_gradient = propagate_correction_gradient
        if ricatti_solver is None:
            self.ricatti_solver = PetkovSolver(tf.constant(10))
        else:
            self.ricatti_solver = ricatti_solver
        super(CorrectedRegularizedTransform, self).__init__(name=name)

    @tf.function
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
        transport_matrix = transport(state.particles, state.log_weights, self.epsilon, self.scaling,
                                     self.convergence_threshold, self.max_iter, state.n_particles)
        weights = state.weights
        transport_correction = self.ricatti_solver(transport_matrix, weights)
        if not self.propagate_correction_gradient:
            transport_correction = tf.stop_gradient(transport_correction)
        res = apply_transport_matrix(state, transport_matrix + transport_correction, flags)
        return res


class PartiallyCorrectedRegularizedTransform(ResamplerBase, metaclass=abc.ABCMeta):
    """Partially Corrected (variance) Regularised Transform - docstring to come."""

    DIFFERENTIABLE = True

    # TODO: Document this really nicely
    def __init__(self, epsilon, scaling=0.75, max_iter=100, convergence_threshold=1e-3,
                 propagate_correction_gradient=True, name='RegularisedTransform'):
        """Constructor

        :param epsilon: float
            Regularizer for Sinkhorn iterates
        :param scaling: float
            Epsilon scaling for sinkhorn iterates
        :param max_iter: int
            max number of iterations in Sinkhorn
        :param convergence_threshold: float
            Fixed point iterates converge when potentials don't move more than this anymore
        :param propagate_correction_gradient: should you propagate the correction factor gradient
        """
        self.convergence_threshold = tf.cast(convergence_threshold, float)
        self.max_iter = tf.cast(max_iter, tf.dtypes.int32)
        self.epsilon = tf.cast(epsilon, float)
        self.scaling = tf.cast(scaling, float)
        self.propagate_correction_gradient = propagate_correction_gradient
        super(PartiallyCorrectedRegularizedTransform, self).__init__(name=name)

    @tf.function
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
        transport_matrix = transport(state.particles, state.log_weights, self.epsilon, self.scaling,
                                     self.convergence_threshold, self.max_iter, state.n_particles)

        weights = tf.expand_dims(state.weights, -1)

        weighted_average = tf.reduce_sum(weights * state.particles, axis=[0, 1], keepdims=True)
        centered_particles = state.particles - weighted_average
        weighted_std = tf.math.sqrt(tf.reduce_sum(weights * centered_particles ** 2, axis=[0, 1], keepdims=True))

        transported_state = apply_transport_matrix(state, transport_matrix, flags)

        transformed_std = tf.math.reduce_std(transported_state.particles, axis=[0, 1], keepdims=True)

        alpha = weighted_std / transformed_std
        beta = (1. - alpha) * weighted_average

        if not self.propagate_correction_gradient:
            alpha = tf.stop_gradient(alpha)
            beta = tf.stop_gradient(beta)

        return attr.evolve(transported_state, particles=alpha * transported_state.particles + beta)
