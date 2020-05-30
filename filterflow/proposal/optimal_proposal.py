import attr
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.proposal.base import ProposalModelBase

tfd = tfp.distributions


class OptimalProposalModel(ProposalModelBase):
    # TODO: should work with any transition function
    """(Locally) optimal proposal.
    See Doucet and al. http://www.irisa.fr/aspi/legland/ref/doucet00b.pdf
    This is only valid for LGSS models.
    """

    def __init__(self, transition_matrix, observation_matrix, transition_covariance_chol, observation_covariance_chol,
                 name='OptimalProposalModel'):
        super(OptimalProposalModel, self).__init__(name=name)
        self._transition_matrix = transition_matrix
        self._observation_matrix = observation_matrix
        self._transition_covariance_chol = transition_covariance_chol
        self._observation_covariance_chol = observation_covariance_chol

        d_x = transition_covariance_chol.shape[0]
        d_y = observation_covariance_chol.shape[0]

        self._transition_covariance_inv = tf.linalg.cholesky_solve(transition_covariance_chol,
                                                                   tf.eye(d_x))

        self._observation_covariance_inv = tf.linalg.cholesky_solve(observation_covariance_chol,
                                                                    tf.eye(d_y))

        self._sigma_inv = self._transition_covariance_inv + tf.linalg.matmul(
            tf.linalg.matmul(observation_matrix,
                             self._observation_covariance_inv,
                             transpose_a=True),
            observation_matrix)

        self._sigma = tf.linalg.inv(self._sigma_inv)
        self._sigma_chol = tf.linalg.cholesky(self._sigma)

    def _get_proposal_dist(self, state, observation):
        mean = tf.linalg.matvec(self._observation_matrix,
                                tf.linalg.matvec(self._observation_covariance_inv,
                                                 observation),
                                transpose_a=True)
        mean += tf.linalg.matvec(self._transition_covariance_inv,
                                 tf.linalg.matvec(self._transition_matrix,
                                                  state.particles))
        mean = tf.linalg.matvec(self._sigma, mean)
        return tfd.MultivariateNormalTriL(mean, self._sigma_chol)

    def propose(self, state: State, inputs: tf.Tensor, observation: tf.Tensor, seed=None):
        """See base class"""
        proposal_dist = self._get_proposal_dist(state, observation)
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
        proposal_dist = self._get_proposal_dist(state, observation)

        return proposal_dist.log_prob(proposed_state.particles)
