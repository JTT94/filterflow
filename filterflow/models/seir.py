import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.observation.base import ObservationSampler
from filterflow.proposal import ProposalModelBase
from filterflow.smc import SMC
from filterflow.transition.base import TransitionModelBase


def split_sei(x):
    S, E, I = tf.split(x, 3, axis=-1)
    return S, E, I


def join_sei(S, E, I):
    x = tf.concat([S, E, I], axis=-1)
    return x


class SEIRTransitionModel(TransitionModelBase):
    def __init__(self, alpha, beta, gamma, log_sig, population_size, name='SEIRTransitionModel'):
        super(SEIRTransitionModel, self).__init__(name=name)

        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.log_sig = log_sig
        self.pop = population_size
        self.normal_dist = tfp.distributions.Normal(loc=tf.constant(0.), scale=tf.exp(log_sig))

    def _loglikelihood(self, prior_state: State, proposed_state: State):
        St_1, Et_1, It_1 = split_sei(prior_state.particles)
        St, Et, It = split_sei(proposed_state.particles)

        eps = (It - (1. - self.gamma) * It_1 - self.alpha * Et_1) / It_1
        log_prob = tf.reduce_sum(self.normal_dist.log_prob(eps), axis=-1)
        return log_prob

    def loglikelihood(self, prior_state: State, proposed_state: State, inputs: tf.Tensor):
        return self._loglikelihood(prior_state, proposed_state)

    def sample(self, state: State, inputs: tf.Tensor, seed=None):
        St_1, Et_1, It_1 = split_sei(state.particles)

        eps = self.normal_dist.sample(seed=seed)

        St = St_1 - self.beta * St_1 * It_1 / self.pop
        Et = (1 - self.alpha) * Et_1 + self.beta * St_1 * It_1 / self.pop

        It = (1 - self.gamma + eps) * It_1 + self.alpha * Et_1

        x = join_sei(St, Et, It)

        return State(particles=x,
                     log_weights=state.log_weights,
                     weights=state.weights,
                     log_likelihoods=state.log_likelihoods)


class SEIRProposalModel(ProposalModelBase):

    def __init__(self, alpha, beta, gamma, log_sig, population_size, name='SEIRProposalModel'):
        super(SEIRProposalModel, self).__init__(name=name)
        self._transition_model = SEIRTransitionModel(alpha, beta, gamma, log_sig, population_size)

    def loglikelihood(self, proposed_state: State, state: State, inputs: tf.Tensor, observation: tf.Tensor):
        return self._proposal_model.loglikelihood(state, proposed_state)

    def propose(self, state: State, inputs: tf.Tensor, observation: tf.Tensor, seed=None):
        return self._proposal_model.sample(state, inputs, seed)


class SEIRObservationModel(ObservationSampler):

    def __init__(self, delta, batch_shape=50, name='SEIRObservationModel'):
        super(SEIRObservationModel, self).__init__(name=name)
        self.half_norm_dist = tfp.distributions.HalfNormal(scale=tf.constant(5.))
        self.delta = delta
        self.batch_shape = batch_shape

    def get_dist(self, state: State):
        S, E, I = split_sei(state.particles)
        b = state.batch_size
        dt = self.delta * I
        pois_dist = tfp.distributions.Poisson(rate=dt)
        # tf.print('Mean deaths', dt)
        return pois_dist

    def loglikelihood(self, state: State, observation: tf.Tensor):
        pois_dist = self.get_dist(state)
        prob = tf.reduce_sum(pois_dist.prob(observation), axis=-1)

        # tf.print('Obs',observation)
        # tf.print('Prob',prob)
        return prob

    def sample(self, state: State):
        pois_dist = self.get_dist(state)
        d = pois_dist.sample()
        return d


def make_filter(model_kwargs, batch_size, n_particles, resampling_method, resampling_criterion):
    kwargs = ['alpha', 'beta', 'gamma', 'delta', 'log_sig', 'population_size']
    alpha = model_kwargs['alpha']
    beta = model_kwargs['beta']
    gamma = model_kwargs['gamma']
    delta = model_kwargs['delta']
    log_sig = model_kwargs['log_sig']
    population_size = model_kwargs['population_size']

    # set trainable variables
    init_values = [alpha, beta, gamma]
    learnable_alpha = tf.Variable(alpha, trainable=True)
    learnable_beta = tf.Variable(beta, trainable=True)
    learnable_gamma = tf.Variable(gamma, trainable=True)
    gradient_variables = [learnable_alpha, learnable_beta, learnable_gamma]

    delta = delta
    log_sig = log_sig

    # init particles
    S0 = population_size - 1.
    I0 = 1.
    E0 = 0.

    initial_sei = tf.constant([S0, E0, I0])
    initial_sei = tf.reshape(initial_sei, [1, 1, 3])
    initial_particles = tf.tile(initial_sei, [batch_size, n_particles, 1])
    initial_particles = tf.cast(initial_particles, dtype=float)

    initial_weights = tf.ones((batch_size, n_particles), dtype=float) / tf.cast(n_particles, float)
    log_likelihoods = tf.zeros(batch_size, dtype=float)
    initial_state = State(particles=initial_particles,
                          log_weights=tf.math.log(initial_weights),
                          weights=initial_weights,
                          log_likelihoods=log_likelihoods)
    # make filter
    transition_model = SEIRTransitionModel(learnable_alpha, learnable_beta, learnable_gamma, log_sig, population_size)
    proposal_model = SEIRProposalModel(learnable_alpha, learnable_beta, learnable_gamma, log_sig, population_size)
    observation_model = SEIRObservationModel(delta)
    smc = SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)

    return initial_state, init_values, gradient_variables, smc
