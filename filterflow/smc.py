import attr
import tensorflow as tf

from filterflow.base import State, Module, DTYPE_TO_STATE_SERIES
from filterflow.constants import MIN_RELATIVE_LOG_WEIGHT
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

    @tf.function
    def update(self, state: State, observation: tf.Tensor, inputs: tf.Tensor):
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
        new_state = self._resampling_correction_term(resampling_flag, new_state, state, observation, inputs)
        return new_state

    def _resampling_correction_term(self, resampling_flag: tf.Tensor, new_state: State, prior_state: State,
                                    observation: tf.Tensor, inputs: tf.Tensor):
        b, n = prior_state.batch_size, prior_state.n_particles
        uniform_log_weights = tf.zeros([b, n]) - tf.math.log(tf.cast(n, float))
        baseline_state = self.propose_and_weight(attr.evolve(prior_state,
                                                             log_weights=uniform_log_weights,
                                                             weights=tf.exp(uniform_log_weights)),
                                                 observation,
                                                 inputs)
        float_flag = tf.cast(resampling_flag, float)
        centered_reward = tf.reshape(float_flag * (new_state.log_likelihoods - baseline_state.log_likelihoods), [-1, 1])
        resampling_correction = prior_state.resampling_correction + tf.reduce_mean(
            tf.stop_gradient(centered_reward) * prior_state.log_weights, 1)
        return attr.evolve(new_state, resampling_correction=resampling_correction)

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
        float_n_particles = tf.cast(state.n_particles, float)
        proposed_state = self._proposal_model.propose(state, inputs, observation)
        log_weights = self._transition_model.loglikelihood(state, proposed_state, inputs)
        log_weights = log_weights + self._observation_model.loglikelihood(proposed_state, observation)
        log_weights = log_weights - self._proposal_model.loglikelihood(proposed_state, state, inputs, observation)
        log_weights = log_weights + state.log_weights
        log_likelihood_increment = tf.math.reduce_logsumexp(log_weights, 1)
        log_likelihoods = state.log_likelihoods + log_likelihood_increment
        normalized_log_weights = normalize(log_weights, 1, True)
        normalized_log_weights = tf.clip_by_value(normalized_log_weights,
                                                  MIN_RELATIVE_LOG_WEIGHT * float_n_particles,
                                                  tf.constant(float('inf')))
        normalized_log_weights = normalize(normalized_log_weights, 1, True)
        return attr.evolve(proposed_state, weights=tf.math.exp(normalized_log_weights),
                           log_weights=normalized_log_weights, log_likelihoods=log_likelihoods)

    @tf.function
    def _return(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor):
        # infer dtype
        dtype = initial_state.particles.dtype

        # init series
        StateSeriesKlass = DTYPE_TO_STATE_SERIES[dtype]
        states_series = StateSeriesKlass(batch_size=initial_state.batch_size,
                                         n_particles=initial_state.n_particles,
                                         dimension=initial_state.dimension)

        data_iterator = iter(observation_series)

        def body(state, states, i):
            observation = data_iterator.get_next()
            state = self.update(state, observation, tf.constant(0.))
            states = states.write(i, state)
            return state, states, i + 1

        def cond(_state, _states, i):
            return i < n_observations

        i0 = tf.constant(0)
        final_state, states, _ = tf.while_loop(cond, body, [initial_state, states_series, i0], )
        return final_state, states_series.stack()

    @tf.function
    def _return_all_loop(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor):
        _, states_series = self._return(initial_state, observation_series, n_observations)
        return states_series

    @tf.function
    def _return_final_loop(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor):
        final_state, states_series = self._return(initial_state, observation_series, n_observations)
        return final_state

    @tf.function
    def __call__(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor,
                 return_final=False):
        """
        :param initial_state: State
            initial state of the filter
        :param observation_series: ObservationSeries
            sequence of observation objects
        :return: tensor array of states
        """
        if return_final:
            return self._return_final_loop(initial_state, observation_series, n_observations)
        else:
            return self._return_all_loop(initial_state, observation_series, n_observations)
