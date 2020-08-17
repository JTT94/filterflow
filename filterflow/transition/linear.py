import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.transition.base import TransitionModelBase


class LinearTransitionModel1d(TransitionModelBase):
    def __init__(self, scalar: tf.Tensor, add_term: tf.Tensor, noise: tfp.distributions.Distribution, name='RandomWalkModel'):
        super(LinearTransitionModel1d, self).__init__(name=name)
        self._scalar = scalar
        self._add_term = add_term
        self._noise = noise

    def push_particles(self, particles):
        pushed_particles =  self._scalar * particles
        pushed_particles = pushed_particles + self._add_term
        return pushed_particles

    def loglikelihood(self, prior_state: State, proposed_state: State, inputs: tf.Tensor):
        """Computes the loglikelihood of an observation given proposed particles
        :param prior_state: State
            State at t-1
        :param proposed_state: State
            Some proposed State for which we want the likelihood given previous state
        :param inputs: tf.Tensor
            Input for transition model
        :return: a tensor of loglikelihoods for all particles in proposed state
        :rtype: tf.Tensor
        """
        batch_size, n_particles, dim = prior_state.particles.shape
        pushed_particles = self.push_particles(prior_state.particles)
        diff = proposed_state.particles - pushed_particles
        log_prob = self._noise.log_prob(diff)
        return tf.reshape(log_prob, [batch_size, n_particles])

    def sample(self, state: State, inputs: tf.Tensor, seed=None):
        """Samples a new proposed state conditionally on prior state and some inputs
        :param state: State
            State of the filter at t-1
        :param inputs: tf.Tensor
            Input for transition model
        :param seed: tf.Tensor
            Seed
        :return: proposed State
        :rtype: State
        """
        pushed_particles = self.push_particles(state.particles)
        res = pushed_particles + self._noise.sample([state.batch_size, state.n_particles], seed=seed)
        return res

class LinearTransitionModel(TransitionModelBase):
    def __init__(self, scalar_matrix: tf.Tensor, add_term: tf.Tensor, noise: tfp.distributions.Distribution, name='RandomWalkModel'):
        super(LinearTransitionModel, self).__init__(name=name)
        self._scalar_matrix = scalar_matrix
        self._add_term = add_term
        self._noise = noise

    def push_particles(self, particles):
        pushed_particles = tf.linalg.matvec(self._scalar_matrix, particles) 
        pushed_particles = pushed_particles + self._add_term
        return pushed_particles

    def loglikelihood(self, prior_state: State, proposed_state: State, inputs: tf.Tensor):
        """Computes the loglikelihood of an observation given proposed particles
        :param prior_state: State
            State at t-1
        :param proposed_state: State
            Some proposed State for which we want the likelihood given previous state
        :param inputs: tf.Tensor
            Input for transition model
        :return: a tensor of loglikelihoods for all particles in proposed state
        :rtype: tf.Tensor
        """
        pushed_particles = self.push_particles(prior_state.particles)
        diff = proposed_state.particles - pushed_particles
        return self._noise.log_prob(diff)

    def sample(self, state: State, inputs: tf.Tensor):
        """Samples a new proposed state conditionally on prior state and some inputs
        :param state: State
            State of the filter at t-1
        :param inputs: tf.Tensor
            Input for transition model
        :return: proposed State
        :rtype: State
        """
        pushed_particles = self.push_particles(state.particles)
        res = pushed_particles + self._noise.sample([state.batch_size, state.n_particles])
        return res
