import abc

import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.univariate.emd_1d import transport_1d


class SVDSlicedTransform(ResamplerBase, metaclass=abc.ABCMeta):
    """Regularised Transform - docstring to come."""
    DIFFERENTIABLE = True

    # TODO: Document this really nicely
    def __init__(self, n_components, name='SVDSlicedTransform'):
        """Constructor

        :param n_components: int
            number of components for the SVD
        """
        self.n_components = tf.cast(n_components, tf.int32)
        super(SVDSlicedTransform, self).__init__(name=name)

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
        b, n, d = state.batch_size, state.n_particles, state.dimension
        float_n_particles = tf.cast(n, float)

        _, _, v = tf.linalg.svd(state.particles)
        v_n = v[..., : self.n_components, :]
        projected_particles = tf.linalg.matmul(v_n, state.particles, transpose_b=True)  # b x n_components x n
        weights = tf.tile(tf.expand_dims(state.weights, 1), [1, self.n_components, 1])  # b x n_components x n

        def preprocess(tensor):
            return tf.tile(tf.expand_dims(tensor, 1), [1, self.n_components, 1, 1])

        def reduce(tensor):
            return tf.reduce_mean(tensor, 1)

        state_particles = preprocess(state.particles)

        additional_state_variables = []
        for variable_name in state.ADDITIONAL_STATE_VARIABLES:
            additional_state = preprocess(getattr(state, variable_name))
            additional_state_variables.append(additional_state)

        _, new_state_particles, *new_additional_states = transport_1d(weights, projected_particles,
                                                                      state_particles, *additional_state_variables)

        new_additional_states_dict = {name: float_n_particles * reduce(v) for name, v in
                                      zip(state.ADDITIONAL_STATE_VARIABLES, new_additional_states)}

        uniform_log_weights = -tf.math.log(float_n_particles) * tf.ones_like(state.log_weights)
        uniform_weights = tf.ones_like(state.weights) / float_n_particles

        return attr.evolve(state, particles=float_n_particles * reduce(new_state_particles),
                           log_weights=uniform_log_weights,
                           weights=uniform_weights, **new_additional_states_dict)
