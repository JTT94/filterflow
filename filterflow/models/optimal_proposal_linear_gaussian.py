import attr
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.observation.linear import LinearObservationModel
from filterflow.proposal import ProposalModelBase
from filterflow.proposal.optimal_proposal import OptimalProposalModel
from filterflow.smc import SMC
from filterflow.transition.random_walk import RandomWalkModel

tfd = tfp.distributions


class LearnableProposalModel(ProposalModelBase):
    """See our paper"""

    def __init__(self, transition_matrix, log_phi_x, phi_y, name='LearnableProposalModel'):
        super(LearnableProposalModel, self).__init__(name=name)
        self._transition_matrix = transition_matrix

        d_x = transition_matrix.shape[0]

        self._log_phi_x = log_phi_x
        self._phi_y = phi_y

    def _get_dist(self, state, observation):
        diag_x_2 = tf.exp(self._log_phi_x / 2.)

        inverse_diag_x = tf.linalg.diag(tf.exp(-self._log_phi_x))

        diag_y = tf.linalg.diag(self._phi_y, num_rows=state.dimension)

        mean = tf.linalg.matvec(self._transition_matrix, state.particles)
        mean += tf.linalg.matvec(diag_y, observation)
        mean = tf.linalg.matvec(inverse_diag_x, mean)
        return tfd.MultivariateNormalDiag(mean, diag_x_2)

    @tf.function
    def propose(self, state: State, inputs, observation: tf.Tensor, seed=None):
        """See base class"""
        proposal_dist = self._get_dist(state, observation)
        proposed_particles = proposal_dist.sample(seed=seed)
        return attr.evolve(state, particles=proposed_particles)

    def loglikelihood(self, proposed_state: State, state: State, inputs: tf.Tensor, observation: tf.Tensor):
        """Interface method for particle proposal
        :param proposed_state: State
            proposed state
        :param state: State
            previous particle filter state
        :param inputs: tf.Tensor
            Control variables (time elapsed, some environment variables, etc)
        :param observation: tf.Tensor
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: tf.Tensor
        """
        proposal_dist = self._get_dist(state, observation)

        return proposal_dist.log_prob(proposed_state.particles)


#
# class LearnableProposalModel(ProposalModelBase):
#     """See our paper"""
#
#     def __init__(self, transition_matrix, log_phi_x, phi_y, name='LearnableProposalModel'):
#         super(LearnableProposalModel, self).__init__(name=name)
#         self._transition_matrix = transition_matrix
#
#         d_x = transition_matrix.shape[0]
#
#         self._log_phi_x = log_phi_x
#         self._phi_y = phi_y
#
#         self._diag_x_2 = tf.exp(tf.linalg.diag(log_phi_x / 2))
#
#         self._inverse_diag_x = tf.exp(tf.linalg.diag(-log_phi_x))
#
#         self._diag_y = tf.linalg.diag(phi_y, num_rows=d_x)
#
#     def _get_dist(self, state, observation):
#         mean = tf.linalg.matvec(self._transition_matrix, state.particles)
#         mean += tf.linalg.matvec(self._diag_y, observation)
#         mean = tf.linalg.matvec(self._inverse_diag_x, mean)
#
#         return tfd.MultivariateNormalDiag(mean, self._diag_x_2)
#
#     @tf.function
#     def propose(self, state: State, inputs, observation: tf.Tensor, seed=None):
#         """See base class"""
#         mu_t = inputs[0]
#         beta_t = inputs[1]
#         sigma_t = inputs[2]
#         transition_matrix = tf.linalg.matmul(tf.linalg.diag(beta_t), self._transition_matrix)
#
#         pushed_particles = tf.reshape(mu_t, [1, 1, -1]) + tf.linalg.matvec(transition_matrix, state.particles)
#
#         scale = tfp.bijectors.ScaleMatvecDiag(sigma_t)
#         scaled_rv = tfd.TransformedDistribution(self._standard_noise, bijector=scale)
#         proposed_particles = pushed_particles + scaled_rv.sample([state.batch_size, state.n_particles], seed=seed)
#         return attr.evolve(state, particles=proposed_particles)
#
#     def loglikelihood(self, proposed_state: State, state: State, inputs: tf.Tensor, observation: tf.Tensor):
#         """Interface method for particle proposal
#         :param proposed_state: State
#             proposed state
#         :param state: State
#             previous particle filter state
#         :param inputs: tf.Tensor
#             Control variables (time elapsed, some environment variables, etc)
#         :param observation: tf.Tensor
#             Look ahead observation for adapted particle proposal
#         :return: proposed State
#         :rtype: tf.Tensor
#         """
#         mu_t, beta_t, sigma_t = inputs
#         transition_matrix = tf.linalg.matmul(tf.linalg.diag(beta_t), self._transition_matrix)
#         pushed_particles = tf.reshape(mu_t, [1, 1, -1]) + tf.linalg.matvec(transition_matrix, state.particles)
#
#         scale = tfp.bijectors.ScaleMatvecDiag(sigma_t)
#         scaled_rv = tfd.TransformedDistribution(self._standard_noise, bijector=scale)
#
#         diff = (pushed_particles - proposed_state.particles)
#         return scaled_rv.log_prob(diff)


def make_filter(observation_matrix, transition_matrix, observation_error_chol, transition_noise_chol, resampling_method,
                resampling_criterion, log_phi_x, phi_y, observation_error_bias=None, transition_noise_bias=None):
    dy, dx = observation_matrix.shape

    if observation_error_bias is None:
        observation_error_bias = tf.zeros(dy)
    if transition_noise_bias is None:
        transition_noise_bias = tf.zeros(dx)

    observation_error = tfp.distributions.MultivariateNormalTriL(observation_error_bias, observation_error_chol)
    observation_model = LinearObservationModel(observation_matrix, observation_error)

    transition_noise = tfp.distributions.MultivariateNormalTriL(transition_noise_bias, transition_noise_chol)
    transition_model = RandomWalkModel(transition_matrix, transition_noise)
    proposal_model = LearnableProposalModel(transition_matrix, log_phi_x, phi_y)

    return SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)


def make_optimal_filter(observation_matrix, transition_matrix, observation_error_chol, transition_noise_chol,
                        resampling_method, resampling_criterion, observation_error_bias=None,
                        transition_noise_bias=None):
    dy, dx = observation_matrix.shape

    if observation_error_bias is None:
        observation_error_bias = tf.zeros(dy)
    if transition_noise_bias is None:
        transition_noise_bias = tf.zeros(dx)

    observation_error = tfp.distributions.MultivariateNormalTriL(observation_error_bias, observation_error_chol)
    observation_model = LinearObservationModel(observation_matrix, observation_error)

    transition_noise = tfp.distributions.MultivariateNormalTriL(transition_noise_bias, transition_noise_chol)
    transition_model = RandomWalkModel(transition_matrix, transition_noise)
    proposal_model = OptimalProposalModel(transition_matrix, observation_matrix, transition_noise_chol,
                                          observation_error_chol)

    return SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)
