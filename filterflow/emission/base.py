import abc

import tensorflow as tf

from filterflow.base import State, ObservationBase, InputsBase


class EmitterModelBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    @tf.function
    def emit(self, state: State, inputs: InputsBase, observation: ObservationBase):
        """Interface method for particle emitter

        :param state: State
            previous particle filter state
        :param inputs: InputsBase
            Control variables (time elapsed, some environment variables, etc)
        :param observation: ObservationBase
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: State
        """


class BootstrapEmitterModel(EmitterModelBase):
    """Standard bootstrap proposal: directly uses the transition model as a proposal.
    """

    def __init__(self, transition_model):
        self._transition_model = transition_model

    @tf.function
    def emit(self, state: State, inputs: InputsBase, _observation: ObservationBase):
        """See base class"""
        return self._transition_model.sample(state, inputs)
