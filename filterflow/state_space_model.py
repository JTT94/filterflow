import attr
import tensorflow as tf

from filterflow.base import State, ObservationBase, InputsBase, Module, StateSeries, ObservationSeries
from filterflow.observation.base import ObservationModelBase
from filterflow.proposal.base import ProposalModelBase
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.criterion import ResamplingCriterionBase
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
        return self._transition_model.sample(state)

    def sample_observation(self, state: State):
        """Samples a new observation conditionally on latent state
        :param state: State
            State of the filter at t
        :return: observartion 
        :rtype: ObservationBase
        """
        return self._observation_model.sample(state)


    def __call__(self, initial_state: State, n_steps : int):
        """
        :param initial_state: State
            initial state of the filter
        :return: tuple of tensor array of states, tensor array of observations
        :rtype: (StateSeries, ObservationSeries)
        
        """

        # init state
        state = attr.evolve(initial_state)

        # infer dimensions and type
        batch_size, n_particles, dimension = state.particles.shape
        dtype = state.particles.dtype

        # init tensor arrays for recording states and outputs
        states = StateSeries(dtype=dtype, batch_size=batch_size, n_particles=n_particles, dimension=dimension)
        observations = ObservationSeries(dtype, dimension)

        # forward loop
        for t in range(n_steps):
            state = self.sample_state(state)
            observation = self.sample_observation(state)
            
            observations.write(t, observation)
            states.write(t, state)

        return states, observations

