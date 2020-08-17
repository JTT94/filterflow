import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.constants import MIN_RELATIVE_LOG_WEIGHT
from filterflow.observation.base import ObservationModelBase
from filterflow.proposal import ProposalModelBase
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.criterion import ResamplingCriterionBase
from filterflow.smc import SMC
from filterflow.transition.base import TransitionModelBase
from filterflow.utils import normalize


class AuxiliaryParticleFilter(SMC):
    def __init__(self, observation_model: ObservationModelBase, transition_model: TransitionModelBase,
                 proposal_model: ProposalModelBase, resampling_criterion: ResamplingCriterionBase,
                 resampling_method: ResamplerBase, name='AuxiliaryParticleFilter'):
        super(AuxiliaryParticleFilter, self).__init__(observation_model, transition_model,
                                                      proposal_model, resampling_criterion, resampling_method,
                                                      name=name)
        raise NotImplementedError("This is not yet implemented")

    @tf.function
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
