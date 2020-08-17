import abc

import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase, resample
from filterflow.resampling.differentiable.regularized_transport.plan import transport


def apply_transport_matrix(state: State, transport_matrix: tf.Tensor, flags: tf.Tensor):
    float_n_particles = tf.cast(state.n_particles, float)
    transported_particles = tf.linalg.matmul(transport_matrix, state.particles)
    uniform_log_weights = -tf.math.log(float_n_particles) * tf.ones_like(state.log_weights)
    uniform_weights = tf.ones_like(state.weights) / float_n_particles

    resampled_particles = resample(state.particles, transported_particles, flags)
    resampled_weights = resample(state.weights, uniform_weights, flags)
    resampled_log_weights = resample(state.log_weights, uniform_log_weights, flags)

    additional_variables = {}

    for additional_state_variable in state.ADDITIONAL_STATE_VARIABLES:
        state_variable = getattr(state, additional_state_variable)
        transported_state_variable = tf.linalg.matmul(transport_matrix, state.particles)
        additional_variables[additional_state_variable] = resample(state_variable, transported_state_variable, flags)

    return attr.evolve(state, particles=resampled_particles, weights=resampled_weights,
                       log_weights=resampled_log_weights)


class RegularisedTransform(ResamplerBase, metaclass=abc.ABCMeta):
    """Regularised Transform - docstring to come."""
    DIFFERENTIABLE = True

    # TODO: Document this really nicely
    def __init__(self, epsilon, scaling=0.75, max_iter=100, convergence_threshold=1e-3, additional_variables_are_state=False, name='RegularisedTransform',
                 **_kwargs):
        """Constructor

        :param epsilon: float
            Regularizer for Sinkhorn iterates
        :param scaling: float
            Epsilon scaling for sinkhorn iterates
        :param max_iter: int
            max number of iterations in Sinkhorn
        :param convergence_threshold: float
            Fixed point iterates converge when potentials don't move more than this anymore
        :param additional_variables_are_state: bool
            Use the "ADDITIONAL_VARIABLES" flag as part of the state or take the conditional expectation
        """
        self.convergence_threshold = tf.cast(convergence_threshold, float)
        self.max_iter = tf.cast(max_iter, tf.dtypes.int32)
        self.epsilon = tf.cast(epsilon, float)
        self.scaling = tf.cast(scaling, float)
        self.additional_variables_are_state = additional_variables_are_state
        super(RegularisedTransform, self).__init__(name=name)

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
        if self.additional_variables_are_state:
            particles_list = [state.particles]
            for additional_state_variable in state.ADDITIONAL_STATE_VARIABLES:
                particles_list.append(getattr(state, additional_state_variable))
            particles = tf.concat(particles_list, axis=-1)
        else:
            particles = state.particles

        transport_matrix = transport(particles, state.log_weights, self.epsilon, self.scaling,
                                     self.convergence_threshold, self.max_iter, state.n_particles)

        return apply_transport_matrix(state, transport_matrix, flags)
