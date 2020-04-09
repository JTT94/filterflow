import abc

import attr
import tensorflow as tf

from filterflow.base import State, InputsBase, Module


class ProposalModelBase(Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def propose(self, state: State, inputs: InputsBase, observation: tf.Tensor):
        """Interface method for particle proposal

        :param state: State
            previous particle filter state
        :param inputs: InputsBase
            Control variables (time elapsed, some environment variables, etc)
        :param observation: Observation
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: State
        """

    @abc.abstractmethod
    def loglikelihood(self, proposed_state: State, state: State, inputs: InputsBase, observation: tf.Tensor):
        """Interface method for particle proposal
        :param proposed_state: State
            proposed state
        :param state: State
            previous particle filter state
        :param inputs: InputsBase
            Control variables (time elapsed, some environment variables, etc)
        :param observation: Observation
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: tf.Tensor
        """


class BootstrapProposalModel(ProposalModelBase):
    """Standard bootstrap proposal: directly uses the transition model as a proposal.
    """

    def __init__(self, transition_model, name='BootstrapProposalModel'):
        super(BootstrapProposalModel, self).__init__(name=name)
        self._transition_model = transition_model

    def propose(self, state: State, inputs: InputsBase, _observation: tf.Tensor):
        """See base class"""
        proposed_particles = self._transition_model.sample(state, inputs)
        return attr.evolve(state, particles=proposed_particles)

    def loglikelihood(self, proposed_state: State, state: State, inputs: InputsBase, observation: tf.Tensor):
        """Interface method for particle proposal
        :param proposed_state: State
            proposed state
        :param state: State
            previous particle filter state
        :param inputs: InputsBase
            Control variables (time elapsed, some environment variables, etc)
        :param observation: Observation
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: tf.Tensor
        """
        return self._transition_model.loglikelihood(state, proposed_state, inputs)
