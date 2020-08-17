import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.observation.base import ObservationModelBase
from filterflow.proposal import BootstrapProposalModel
from filterflow.smc import SMC
from filterflow.transition.base import TransitionModelBase

tfd = tfp.distributions


class SVTransitionModel(TransitionModelBase):
    """
    X_t = mu + F(X_{t-1} - mu) + U_t,   U_t~N(0,transition_covariance)
    """

    def __init__(self, mu, F, transition_covariance_chol, name='SVTransitionModel'):
        super(SVTransitionModel, self).__init__(name=name)
        self._mu = mu
        self._F = F
        self._transition_covariance_chol = transition_covariance_chol

    def sample(self, state: State, inputs: tf.Tensor, seed=None):
        dist = tfd.MultivariateNormalTriL(self._mu - tf.linalg.matvec(self._F, self._mu),
                                          self._transition_covariance_chol)
        pushed_particles = tf.linalg.matvec(self._F, state.particles)
        res = pushed_particles + dist.sample([state.batch_size, state.n_particles], seed=seed)
        return res

    def loglikelihood(self, prior_state: State, proposed_state: State, inputs: tf.Tensor):
        pushed_particles = tf.linalg.matvec(self._F, prior_state.particles)
        diff = proposed_state.particles - pushed_particles
        dist = tfd.MultivariateNormalTriL(self._mu - tf.linalg.matvec(self._F, self._mu),
                                          self._transition_covariance_chol)
        return dist.log_prob(diff)


class SVObservationModel(ObservationModelBase):

    def __init__(self, observation_covariance_chol, name='SVObservationModel'):
        super(ObservationModelBase, self).__init__(name=name)

        self._dist = tfd.MultivariateNormalTriL(scale_tril=observation_covariance_chol)

    @tf.function
    def loglikelihood(self, state: State, observation: tf.Tensor):
        """Computes the loglikelihood of an observation given proposed particles
        :param state: State
            Proposed (predicted) state of the filter given State at t-1 and Observation
        :param observation: ObservationBase
            User/Process given observation
        :return: a tensor of loglikelihoods for all particles
        :rtype: tf.Tensor
        """
        batch_size, n_particles = state.batch_size, state.n_particles

        exp_state = tf.exp(state.particles / 2)

        scale = tfp.bijectors.ScaleMatvecDiag(exp_state)
        dist = tfd.TransformedDistribution(distribution=self._dist, bijector=scale)

        scaled_observation = tf.reshape(observation, [1, 1, -1])

        log_prob = dist.log_prob(scaled_observation)
        return tf.reshape(log_prob, [batch_size, n_particles])

    @tf.function
    def sample(self, state: State):
        """Samples a new observation conditionally on latent state
        :param state: State
            State of the filter at t
        :return: observartion
        :rtype: Observation
        """
        exp_state = tf.exp(-state.particles / 2)

        scaled_observation = self._dist.sample()

        return tf.reshape(scaled_observation, [1, 1, -1]) * exp_state


def make_filter(mu, F, transition_noise_chol, observation_covariance_chol, resampling_method,
                resampling_criterion):
    observation_model = SVObservationModel(observation_covariance_chol)

    transition_model = SVTransitionModel(mu, F, transition_noise_chol)
    proposal_model = BootstrapProposalModel(transition_model)

    return SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)
