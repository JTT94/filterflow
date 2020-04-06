import attr
import tensorflow as tf

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

    def __call__(self, initial_state: tf.Tensor, n_steps : int):
        """
        :param initial_state: State
            initial state of the filter
        :return: tuple of tensor array of states, tensor array of observations
        :rtype: (StateSeries, ObservationSeries)
        
        """

        # infer dtype
        dtype = initial_state.dtype

        # init particle
        initial_state = self.init_state(initial_state)
        state = attr.evolve(initial_state)
        
        # get observation dim
        test_obs = self.sample_observation(state)
        obs_dim = test_obs.observation.shape[2]
        
        # init tensor arrays for recording states and outputs
         # init series
        states_constructor = DTYPE_TO_STATE_SERIES[dtype]
        states = states_constructor(batch_size=state.batch_size,
                                  n_particles=state.n_particles,
                                  dimension=state.dimension)

        observations_constructor = DTYPE_TO_OBSERVATION_SERIES[dtype]
        observations = observations_constructor(shape=[1,1, obs_dim])
        # forward loop
        for t in range(n_steps):
            state_particle = self.sample_state(state)
            state = attr.evolve(state, particles=state_particle)
            observation = self.sample_observation(state)
            
            observations = observations.write(t, observation)
            states = states.write(t, state)

        return states, observations

