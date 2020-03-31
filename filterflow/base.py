import abc

import attr
import tensorflow as tf


class Module(tf.Module, metaclass=abc.ABCMeta):
    """
    Base class for filterflow Modules - __call__ is ignored
    """

    def __call__(self, *args, **kwargs):
        pass


@attr.s(frozen=True)
class State:
    """Particle Filter State
    State encapsulates the information about the particle filter current state.
    """
    batch_size = attr.ib()
    n_particles = attr.ib()
    dimension = attr.ib()

    particles = attr.ib()
    log_weights = attr.ib()
    weights = attr.ib()
    log_likelihoods = attr.ib()

@attr.s
class StateSeries():
    dtype = attr.ib()
    batch_size = attr.ib()
    n_particles = attr.ib()
    dimension = attr.ib()
    
    particles_ta = attr.ib(init=False) 
    log_weights_ta = attr.ib(init=False)
    weights_ta = attr.ib(init=False)
    log_likelihoods_ta = attr.ib(init=False)
        
        # init series
    def __attrs_post_init__(self):
        # particles
        self.particles_ta = tf.TensorArray(self.dtype, 
                                      size=0, dynamic_size=True, clear_after_read=False,
                                      element_shape = tf.TensorShape([self.batch_size,
                                                                      self.n_particles,
                                                                      self.dimension]))
        # log_weights
        self.log_weights_ta = tf.TensorArray(self.dtype, 
                                        size=0, dynamic_size=True, clear_after_read=False,
                                        element_shape = tf.TensorShape([self.batch_size,
                                                                        self.n_particles]))
        # weights
        self.weights_ta = tf.TensorArray(self.dtype, 
                                    size=0, dynamic_size=True, clear_after_read=False,
                                    element_shape = tf.TensorShape([self.batch_size,
                                                                    self.n_particles]))
        # log_likelihoods
        self.log_likelihoods_ta = tf.TensorArray(self.dtype, 
                                            size=0, dynamic_size=True, clear_after_read=False,
                                            element_shape = tf.TensorShape([1,]))
    
    def write(self, t, state):
        self.particles_ta = self.particles_ta.write(t, state.particles)
        self.log_weights_ta = self.log_weights_ta.write(t, state.log_weights)
        self.weights_ta = self.weights_ta.write(t, state.weights)
        self.log_likelihoods_ta = self.log_likelihoods_ta.write(t, state.log_likelihoods)
    
    def read(self, t):
        particles = self.particles_ta.read(t)
        log_weights = self.log_weights_ta.read(t)
        weights = self.weights_ta.read(t)
        log_likelihoods = self.log_likelihoods_ta.read(t)


        state = State(batch_size = self.batch_size,
                     n_particles = self.n_particles,
                     dimension = self.dimension,
                     particles = particles, 
                     log_weights= log_weights,
                     weights=weights, 
                     log_likelihoods=log_likelihoods)
        return state

@attr.s
class ObservationBase(metaclass=abc.ABCMeta):

    observation = attr.ib()
    shape = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.shape = self.observation.shape

class ObservationSeries(metaclass=abc.ABCMeta):
    dtype = attr.ib()
    dimension = attr.ib()
    n_observations = attr.ib(init=False)
    
    def __attrs_post_init__(self):
        self.n_observations = 0

    @abc.abstractmethod
    def write(self, t, observation):
        pass
    
    @abc.abstractmethod
    def read(self, t):
        pass

@attr.s
class InputsBase(metaclass=abc.ABCMeta):
    """Container interface to implement an arbitrary collection of inputs to give to the filter at time of prediction
    """
