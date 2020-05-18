import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.observation.base import ObservationModelBase
from filterflow.proposal import BootstrapProposalModel
from filterflow.smc import SMC
from filterflow.transition.random_walk import RandomWalkModel

tfd = tfp.distributions


class SVObservationModel(ObservationModelBase):

    def __init__(self, B, log_psi, name='SVObservationModel'):
        super(ObservationModelBase, self).__init__(name=name)

        self.B = B
        self.log_psi = log_psi
        self.M = B.shape[0]

    def generate_dist(self, state):
        alpha = tf.clip_by_value(state.particles, -6., 0.)
        e_alpha_2 = tf.exp(alpha / 2.)
        e_alpha_2 = tf.reshape(e_alpha_2, [state.batch_size, state.n_particles, 1, state.dimension])
        B_tilde = tf.reshape(self.B, [1, 1, self.B.shape[0], self.B.shape[1]]) * e_alpha_2
        Cov = tf.linalg.matmul(B_tilde, B_tilde, transpose_b=True)
        Psi = tf.reshape(tf.linalg.diag(tf.exp(self.log_psi)), [1, 1, self.M, self.M])
        cov_chol = tf.linalg.cholesky(Cov + Psi, name='choleski')
        obs_dist = tfd.MultivariateNormalTriL(tf.zeros(self.M, dtype=float), cov_chol,
                                              name='distribution')

        return obs_dist

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
        obs_dist = self.generate_dist(state)
        log_prob = obs_dist.log_prob(observation)
        return tf.reshape(log_prob, [batch_size, n_particles])

    @tf.function
    def sample(self, state: State):
        """Samples a new observation conditionally on latent state
        :param state: State
            State of the filter at t
        :return: observartion
        :rtype: Observation
        """
        obs_dist = self.generate_dist(state)
        observation = obs_dist.sample()

        return observation


def make_filter(observation_matrix, transition_matrix, log_psi, B, transition_noise_chol, resampling_method,
                resampling_criterion, transition_noise_bias=None):
    dy, dx = observation_matrix

    if transition_noise_bias is None:
        transition_noise_bias = tf.zeros(dx)

    observation_model = SVObservationModel(B, log_psi)

    transition_noise = tfp.distributions.MultivariateNormalTriL(transition_noise_bias, transition_noise_chol)
    transition_model = RandomWalkModel(transition_matrix, transition_noise)
    proposal_model = BootstrapProposalModel(transition_model)

    return SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)
