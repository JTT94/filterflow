import attr
import tensorflow as tf

from filterflow.base import State, Observation, InputsBase, Module, StateSeries, ObservationSeries
from filterflow.observation.base import ObservationModelBase
from filterflow.proposal.base import ProposalModelBase
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.criterion import ResamplingCriterionBase
from filterflow.transition.base import TransitionModelBase
from filterflow.utils import normalize


class SMC(Module):
    def __init__(self, observation_model: ObservationModelBase, transition_model: TransitionModelBase,
                 proposal_model: ProposalModelBase, resampling_criterion: ResamplingCriterionBase,
                 resampling_method: ResamplerBase, name='SMC'):
        super(SMC, self).__init__(name=name)
        self._observation_model = observation_model
        self._transition_model = transition_model
        self._proposal_model = proposal_model
        self._resampling_criterion = resampling_criterion
        self._resampling_method = resampling_method

    def predict(self, state: State, inputs: InputsBase):
        """Predict step of the filter

        :param state: State
            prior state of the filter
        :param inputs: InputsBase
            Inputs used for preduction
        :return: Predicted State
        :rtype: State
        """
        return self._transition_model.sample(state, inputs)
    
    def update(self, state: State, observation: Observation,
               inputs: InputsBase):
        """
        :param state: State
            current state of the filter
        :param observation: Observation
            observation to compare the state against
        :param inputs: InputsBase
            inputs for the observation_model
        :return: Updated weights
        """
        # check for if resampling is required
        resampling_flag = self._resampling_criterion.apply(state)
        # perform resampling
        resampled_state = self._resampling_method.apply(state, resampling_flag)
        # perform sequential IS step
        new_state = self.propose_and_weight(resampled_state, observation, inputs)

        return new_state

    def propose_and_weight(self, state: State, observation: Observation,
                           inputs: InputsBase):
        """
        :param state: State
            current state of the filter
        :param observation: Observation
            observation to compare the state against
        :param inputs: InputsBase
            inputs for the observation_model
        :return: Updated weights
        """
        
        proposed_state = self._proposal_model.propose(state, inputs, observation)
        log_weights = self._transition_model.loglikelihood(state, proposed_state, inputs)
        log_weights = log_weights + self._observation_model.loglikelihood(proposed_state, observation)
        log_weights = log_weights - self._proposal_model.loglikelihood(proposed_state, state, inputs, observation)
        log_weights = log_weights + state.log_weights

        log_likelihood_increment = tf.math.reduce_logsumexp(log_weights, 1)
        log_likelihoods = state.log_likelihoods + log_likelihood_increment
        normalized_log_weights = normalize(log_weights, 1, True)
        return attr.evolve(proposed_state, weights=tf.math.exp(normalized_log_weights),
                           log_weights=normalized_log_weights, log_likelihoods=log_likelihoods)


    def __call__(self, initial_state: State, observation_series : ObservationSeries, return_final = False):
        """
        :param initial_state: State
            initial state of the filter
        :param observation_series: ObservationSeries
            sequence of observation objects
        :return: tensor array of states
        """

        # init state
        state = attr.evolve(initial_state)

        # infer dimensions and type
        batch_size, n_particles, dimension = state.particles.shape
        dtype = state.particles.dtype

        # init series
        states = StateSeries(dtype=dtype, batch_size=batch_size, n_particles=n_particles, dimension=dimension)

        # forward loop
        for t in range(observation_series.n_observations):
            observation = observation_series.read(t)
            state = self.update(state, observation, None)
            states.write(t, state)
        return states

