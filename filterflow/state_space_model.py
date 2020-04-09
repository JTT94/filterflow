import attr
import tensorflow as tf
import numpy as np
from filterflow.base import State, Observation, InputsBase, Module, StateSeries, DTYPE_TO_OBSERVATION_SERIES, DTYPE_TO_STATE_SERIES
from filterflow.observation.base import ObservationSampler
from filterflow.transition.base import TransitionModelBase
from filterflow.utils import normalize


class StateSpaceModel(Module):
    def __init__(self, observation_model: ObservationSampler, transition_model: TransitionModelBase, name='StateSpaceModel'):
        super(StateSpaceModel, self).__init__(name=name)
        self._observation_model = observation_model
        self._transition_model = transition_model


    def sample_state(self, state: State):
        """Apply transition on latent state"

        :param state: State
            prior state of the filter
        :return: Predicted State
        :rtype: State
        """
        return self._transition_model.sample(state, None)

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
        dim = state_value.shape[0]
        initial_particle = tf.reshape(state_value, [1, 1, dim])

        # create state object with 1 batch and 1 particle
        weights = tf.ones((1, 1), dtype=dtype )
        log_likelihoods = tf.zeros((1), dtype=dtype)
        initial_state = State(initial_particle, 
                            log_weights= tf.math.log(weights),
                            weights=weights, 
                            log_likelihoods=log_likelihoods)
        return initial_state

    def sample(self, state_value: tf.Tensor, n_steps : int):
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
        
        # get observation dim
        test_obs = self.sample_observation(state)
        obs_dim = test_obs.shape[2]
        
        # init tensor arrays for recording states and outputs
         # init series
        states = []
        observations = []
        # forward loop
        for t in range(n_steps):
            state_particle = self.sample_state(state)
            state = attr.evolve(state, particles=state_particle)
            observation = self.sample_observation(state)
            
            observations.append(observation)
            states.append(state)

        return states, observations


    def __call__(self, state_value: tf.Tensor, n_steps : int):
        states, observations = self.sample(state_value, n_steps)
        
        return states, observations
