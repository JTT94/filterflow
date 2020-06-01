import attr
import tensorflow as tf
from tensorflow_probability.python.internal import samplers

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

    @tf.function
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
    def update(self, state: State, observation: tf.Tensor, inputs: tf.Tensor, seed1=None, seed2=None):
        """
        :param state: State
            current state of the filter
        :param observation: tf.Tensor
            observation to compare the state against
        :param inputs: tf.Tensor
            inputs for the observation_model
        :return: Updated weights
        """
        t = state.t
        float_t = tf.cast(t, tf.float32)
        float_t_1 = float_t + 1.
        if seed1 is None or seed2 is None:
            temp_seed = tf.random.uniform((), 0, 2 ** 16, tf.int32)
            seed1, seed2 = samplers.split_seed(temp_seed, n=2, salt='propose_and_weight')
        # check if resampling is required
        # tf.print("weights", tf.reduce_min(state.log_weights, 1))
        # tf.print("ess_1", 1 / tf.reduce_sum(state.weights ** 2, -1))
        resampling_flag, ess = self._resampling_criterion.apply(state)
        # tf.print("ess", ess)
        # update running average efficient sample size
        state = attr.evolve(state, ess=ess / float_t_1 + state.ess * (float_t / float_t_1))
        # perform resampling
        resampled_state = self._resampling_method.apply(state, resampling_flag, seed1)
        # perform sequential IS step
        new_state = self.propose_and_weight(resampled_state, observation, inputs, seed2)
        new_state = self._resampling_correction_term(resampling_flag, new_state, state, observation, inputs)
        # increment t
        return attr.evolve(new_state, t=t + 1)

    @tf.function
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
                           inputs: tf.Tensor, seed=None):
        """
        :param state: State
            current state of the filter
        :param observation: tf.Tensor
            observation to compare the state against
        :param inputs: tf.Tensor
            inputs for the observation_model
        :return: Updated weights
        """
        proposed_state = self._proposal_model.propose(state, inputs, observation, seed=seed)
        observation_log_likelihoods = self._observation_model.loglikelihood(proposed_state, observation)

        log_weights = self._transition_model.loglikelihood(state, proposed_state, inputs)
        log_weights = log_weights + observation_log_likelihoods
        log_weights = log_weights - self._proposal_model.loglikelihood(proposed_state, state, inputs, observation)
        log_weights = log_weights + state.log_weights

        log_likelihood_increment = tf.math.reduce_logsumexp(log_weights, 1)
        log_likelihoods = state.log_likelihoods + log_likelihood_increment
        normalized_log_weights = normalize(log_weights, 1, state.n_particles, True)
        return attr.evolve(proposed_state,
                           weights=tf.math.exp(normalized_log_weights),
                           log_weights=normalized_log_weights,
                           log_likelihoods=log_likelihoods)

    @tf.function
    def _return(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor,
                inputs_series: tf.data.Dataset, seed=None):

        if seed is None:
            temp_seed = tf.random.uniform((), 0, 2 ** 16, tf.int32)
            seed, = samplers.split_seed(temp_seed, n=1, salt='propose_and_weight')
            paddings = tf.constant([[0, 0], [0, 0]])
        else:
            paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
        seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))
        # infer dtype
        dtype = initial_state.particles.dtype

        # init series
        StateSeriesKlass = DTYPE_TO_STATE_SERIES[dtype.name]
        states_series = StateSeriesKlass(batch_size=initial_state.batch_size,
                                         n_particles=initial_state.n_particles,
                                         dimension=initial_state.dimension)

        data_iterator = iter(observation_series)
        inputs_iterator = iter(inputs_series)

        def body(state, states, i, seed):
            observation = data_iterator.get_next()
            inputs = inputs_iterator.get_next()
            seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt='update')
            state = self.update(state, observation, inputs, seed1, seed2)
            states = states.write(i, state)
            return state, states, i + 1, seed

        def cond(_state, _states, i, _seed):
            return i < n_observations

        i0 = tf.constant(0)
        final_state, states_series, _, _ = tf.while_loop(cond, body, [initial_state, states_series, i0, seed], )

        return final_state, states_series.stack()

    @tf.function
    def _return_all_loop(self, initial_state: State, observation_series: tf.data.Dataset,
                         n_observations: tf.Tensor, inputs_series: tf.data.Dataset, seed=None):
        _, states_series = self._return(initial_state, observation_series, n_observations, inputs_series, seed)
        return states_series

    @tf.function
    def _return_final_loop(self, initial_state: State, observation_series: tf.data.Dataset,
                           n_observations: tf.Tensor, inputs_series: tf.data.Dataset, seed=None):
        final_state, _ = self._return(initial_state, observation_series, n_observations, inputs_series, seed)
        return final_state

    def __call__(self, initial_state: State, observation_series: tf.data.Dataset, n_observations: tf.Tensor,
                 inputs_series: tf.data.Dataset = None, return_final=False, seed=None):
        """
        :param initial_state: State
            initial state of the filter
        :param observation_series: ObservationSeries
            sequence of observation objects
        :return: tensor array of states
        """
        if inputs_series is None:
            inputs_series = tf.data.Dataset.range(tf.cast(n_observations, tf.int64), output_type=tf.int32)
        if return_final:
            return self._return_final_loop(initial_state, observation_series, n_observations, inputs_series, seed)
        else:
            return self._return_all_loop(initial_state, observation_series, n_observations, inputs_series, seed)
