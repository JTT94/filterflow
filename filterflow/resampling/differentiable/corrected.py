import abc

import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.differentiable.biased import apply_transport_matrix
from filterflow.resampling.differentiable.regularized_transport.plan import transport
from filterflow.resampling.differentiable.ricatti.solver import PetkovSolver
from filterflow.utils import mean, std
from filterflow.resampling.criterion import neff


@tf.function
def _fill_na(x):
    return tf.where(tf.math.is_finite(x), x, 1.)


@tf.function
def _make_alpha(log_weighted_std, log_transformed_std):
    res = _fill_na(tf.exp(log_weighted_std - log_transformed_std))
    return res


class CorrectedRegularizedTransform(ResamplerBase, metaclass=abc.ABCMeta):
    """Corrected Regularised (covariance) Transform - docstring to come."""

    DIFFERENTIABLE = True

    # TODO: Document this really nicely
    def __init__(self, epsilon, scaling=0.75, max_iter=100, convergence_threshold=1e-3, ricatti_solver=None,
                 propagate_correction_gradient=True, name='RegularisedTransform', **_kwargs):
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
            self.ricatti_solver = PetkovSolver(tf.constant(50))
        else:
            self.ricatti_solver = ricatti_solver
        super(CorrectedRegularizedTransform, self).__init__(name=name)

    @tf.function
    def apply(self, state: State, flags: tf.Tensor, seed=None):
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
    def __init__(self, intermediate_resampler: ResamplerBase, name='RegularisedTransform'):
        """Constructor

        :param intermediate_resampler: ResamplerBase
            Intermediate resampling to be corrected
        """
        self.intermediate_resampler = intermediate_resampler
        super(PartiallyCorrectedRegularizedTransform, self).__init__(name=name)

    @tf.function
    def apply(self, state: State, flags: tf.Tensor, seed=None):
        """ Resampling method

        :param state State
            Particle filter state
        :param flags: tf.Tensor
            Flags for resampling
        :return: resampled state
        :rtype: State
        """
        # TODO: The real batch_size is the sum of flags. We shouldn't do more operations than we need...
        resampled_state = self.intermediate_resampler.apply(state, flags)

        _, ess = neff(state.weights, assume_normalized=True, is_log=False, threshold=tf.constant(0.))
        weighted_average = mean(state, is_log=False)
        weighted_std = std(state, weighted_average, is_log=True)
        transformed_std = std(resampled_state, is_log=True)
        alpha = tf.where(tf.reshape(ess > 3., [-1, 1, 1]), _make_alpha(weighted_std, transformed_std), 1.)
        # Otherwise weighted std doesn't mean anything
        beta = (1. - alpha) * weighted_average
        particles = alpha * resampled_state.particles + beta

        particles = tf.stop_gradient(particles - resampled_state.particles) + resampled_state.particles

        new_state = attr.evolve(resampled_state, particles=particles)

        return new_state
