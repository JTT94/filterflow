import tensorflow as tf

from filterflow.base import State, ObservationBase, InputsBase
from filterflow.emission.base import EmitterModelBase
from filterflow.observation.base import ObservationModelBase
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.criterion import ResamplingCriterionBase
from filterflow.transition.base import TransitionModelBase
from filterflow.utils import normalize


class ParticleFilter(tf.Module):

    def __init__(self, observation_model: ObservationModelBase, transition_model: TransitionModelBase,
                 emission_model: EmitterModelBase, resampling_criterion: ResamplingCriterionBase,
                 resampling_method: ResamplerBase, name=None):
        super(ParticleFilter, self).__init__(name=name)
        self._observation_model = observation_model
        self._transition_model = transition_model
        self._emission_model = emission_model
        self._resampling_criterion = resampling_criterion
        self._resampling_method = resampling_method

    def propose(self, state: State, observation: ObservationBase, inputs: InputsBase):
        """Predict step of the filter

        :param state: State
            prior state of the filter
        :param observation: ObservationBase
            Observation used for look ahead proposal
        :param inputs: InputsBase
            Inputs used for prediction
        :return: Proposed State
        :rtype: State
        """
        print(state.particles.shape)
        resampling_flag = self._resampling_criterion.apply(state)
        state = self._resampling_method.apply(state, resampling_flag)
        print(state.particles.shape)
        proposal = self._emission_model.emit(state, inputs, observation)
        return proposal

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

    def update_weights(self, prior_state: State, proposed_state: State, observation: ObservationBase,
                       inputs: InputsBase):
        """
        :param prior_state: State
            prior state of the filter
        :param proposed_state:
            proposed state of the filter to be corrected
        :param observation: ObservationBase
            observation to compare the state against
        :param inputs: InputsBase
            inputs for the observation_model
        :return: Updated weights
        """
        log_weights = self._transition_model.loglikelihood(prior_state, proposed_state, inputs)
        log_weights = log_weights + self._observation_model.loglikelihood(proposed_state, observation)

        log_likelihood_increment = tf.math.reduce_logsumexp(log_weights, 0)
        log_likelihood = prior_state.log_likelihood + log_likelihood_increment

        log_weights = log_weights + prior_state.log_weights
        normalized_log_weights = normalize(log_weights, 0, True)
        return State(proposed_state.n_particles, proposed_state.batch_size, proposed_state.dimension,
                     proposed_state.particles, normalized_log_weights, None, log_likelihood, False)
