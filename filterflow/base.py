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
class ObservationBase(metaclass=abc.ABCMeta):
    observation = attr.ib()
    shape = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.shape = self.observation.shape


@attr.s
class InputsBase(metaclass=abc.ABCMeta):
    """Container interface to implement an arbitrary collection of inputs to give to the filter at time of prediction
    """
