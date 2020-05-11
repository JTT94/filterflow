import attr
import tensorflow as tf

from filterflow.base import State, Module, DTYPE_TO_STATE_SERIES
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

    def predict(self, state: State, inputs: tf.Tensor):
        """Predict step of the filter

        :param state: State
            prior state of the filter
        :param inputs: tf.Tensor
            Inputs used for preduction
        :return: Predicted State
        :rtype: State
        """
        return self._transition_model.sample(state, inputs)

    def update(self, state: State, observation: tf.Tensor,
               inputs: tf.Tensor):
        """
        :param state: State
            current state of the filter
        :param observation: tf.Tensor
            observation to compare the state against
        :param inputs: tf.Tensor
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

    def propose_and_weight(self, state: State, observation: tf.Tensor,
                           inputs: tf.Tensor):
        """
        :param state: State
            current state of the filter
        :param observation: tf.Tensor
            observation to compare the state against
        :param inputs: tf.Tensor
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

    @tf.function
    def _return(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor, inputs_series: tf.data.Dataset):
        # infer dtype
        dtype = initial_state.particles.dtype

        # init series
        StateSeriesKlass = DTYPE_TO_STATE_SERIES[dtype]
        states_series = StateSeriesKlass(batch_size=initial_state.batch_size,
                                         n_particles=initial_state.n_particles,
                                         dimension=initial_state.dimension)

        data_iterator = iter(observation_series)
        inputs_iterator = iter(inputs_series)

        def body(state, states_series, i):
            observation = data_iterator.get_next()
            inputs = inputs_iterator.get_next()
            state = self.update(state, observation, inputs)
            states_series = states_series.write(i, state)
            return state, states_series, i + 1

        def cond(_state, _states_series, i):
            return i < n_observations

        i0 = tf.constant(0)
        final_state, states_series, _ = tf.while_loop(cond, body, [initial_state, states_series, i0], )
        return final_state, states_series.stack()

    @tf.function
    def _return_all_loop(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor, inputs_series: tf.data.Dataset):
        _, states_series = self._return(initial_state, observation_series, n_observations, inputs_series)
        return states_series

    @tf.function
    def _return_final_loop(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor, inputs_series: tf.data.Dataset):
        final_state, _ = self._return(initial_state, observation_series, n_observations, inputs_series)
        return final_state

    def __call__(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor, inputs_series: tf.data.Dataset,
                 return_final=False):
        """
        :param initial_state: State
            initial state of the filter
        :param observation_series: ObservationSeries
            sequence of observation objects
        :return: tensor array of states
        """
        if return_final:
            return self._return_final_loop(initial_state, observation_series, n_observations, inputs_series)
        else:
            return self._return_all_loop(initial_state, observation_series, n_observations, inputs_series)
