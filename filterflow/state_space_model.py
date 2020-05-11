import attr
import tensorflow as tf

from filterflow.base import State, Module
from filterflow.observation.base import ObservationSampler
from filterflow.transition.base import TransitionModelBase


class StateSpaceModel(Module):
    def __init__(self, observation_model: ObservationSampler, transition_model: TransitionModelBase,
                 name='StateSpaceModel'):
        super(StateSpaceModel, self).__init__(name=name)
        self._observation_model = observation_model
        self._transition_model = transition_model

    @tf.function
    def sample_state(self, state: State):
        """Apply transition on latent state"

        :param state: State
            prior state of the filter
        :return: Predicted State
        :rtype: State
        """
        return self._transition_model.sample(state, tf.constant(0.))

    @tf.function
    def sample_observation(self, state: State):
        """Samples a new observation conditionally on latent state
        :param state: State
            State of the filter at t
        :return: observartion 
        :rtype: ObservationBase
        """
        return self._observation_model.sample(state)

    def init_state(self, state_value):
        dtype = state_value.dtype

        if len(state_value.shape) > 0:
            dim = state_value.shape[0]
        else:
            dim = tf.size(state_value).numpy()

        initial_particle = tf.reshape(state_value, [1, 1, dim])

        # create state object with 1 batch and 1 particle
        weights = tf.ones((1, 1), dtype=dtype)
        log_likelihoods = tf.zeros((1), dtype=dtype)
        initial_state = State(initial_particle,
                              log_weights=tf.math.log(weights),
                              weights=weights,
                              log_likelihoods=log_likelihoods,
                              ancestor_indices=None,
                              resampling_correction=None)
        return initial_state

    @tf.function
    def sample(self, state_value: tf.Tensor, n_steps: int):
        """
        :param state_value: Tensor
            initial state of the filter
        :return: tuple of tensor array of states, tensor array of observations
        :rtype: (StateSeries, ObservationSeries)
        
        """

        # infer dtype
        dtype = state_value.dtype

        # init particle
        initial_state = self.init_state(state_value)
        state = attr.evolve(initial_state)

        observations = tf.TensorArray(dtype=dtype, size=n_steps)
        states = tf.TensorArray(dtype=dtype, size=n_steps)

        # forward loop
        for t in tf.range(n_steps):
            state_particle = self.sample_state(state)
            state = attr.evolve(state, particles=state_particle)
            observation = self.sample_observation(state)

            observations = observations.write(t, observation)
            states = states.write(t, state_particle)

        return states.stack(), observations.stack()

    @tf.function
    def __call__(self, state_value: tf.Tensor, n_steps: int):
        states, observations = self.sample(state_value, n_steps)
        return states, observations
