import tensorflow as tf

from filterflow.base import State, ObservationBase, InputsBase
from filterflow.observation.base import ObservationModelBase
from filterflow.proposal.base import ProposalModelBase
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.criterion import ResamplingCriterionBase
from filterflow.transition.base import TransitionModelBase
from filterflow.utils import normalize


class ParticleFilter(object):

    def __init__(self, observation_model: ObservationModelBase, transition_model: TransitionModelBase,
                 proposal_model: ProposalModelBase, resampling_criterion: ResamplingCriterionBase,
                 resampling_method: ResamplerBase, name=None):
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

    def propose_and_update(self, state: State, observation: ObservationBase,
                           inputs: InputsBase):
        """
        :param state: State
            current state of the filter
        :param observation: ObservationBase
            observation to compare the state against
        :param inputs: InputsBase
            inputs for the observation_model
        :return: Updated weights
        """
        resampling_flag = self._resampling_criterion.apply(state)
        resampled_state = self._resampling_method.apply(state, resampling_flag)
        proposed_state = self._proposal_model.propose(resampled_state, inputs, observation)

        log_weights = self._transition_model.loglikelihood(state, proposed_state, inputs)
        log_weights = log_weights + self._observation_model.loglikelihood(proposed_state, observation)
        log_weights = log_weights - self._proposal_model.loglikelihood(proposed_state, state, inputs, observation)
        log_likelihood_increment = tf.math.reduce_logsumexp(log_weights, 1)
        log_likelihoods = state.log_likelihoods + log_likelihood_increment

        log_weights = log_weights + resampled_state.log_weights
        normalized_log_weights = normalize(log_weights, 1, True)
        return State(batch_size = proposed_state.batch_size, 
                    n_particles = proposed_state.n_particles, 
                    dimension   = proposed_state.dimension, 
                    particles   = proposed_state.particles,
                    log_weights = normalized_log_weights, 
                    weights     = None, 
                    log_likelihoods = log_likelihoods,
                    check_shapes    = state.check_shapes)
        