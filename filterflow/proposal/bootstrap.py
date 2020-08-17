import attr
import tensorflow as tf

from filterflow.base import State
from filterflow.proposal.base import ProposalModelBase


class BootstrapProposalModel(ProposalModelBase):
    """Standard bootstrap proposal: directly uses the transition model as a proposal.
    """

    def __init__(self, transition_model, name='BootstrapProposalModel'):
        super(BootstrapProposalModel, self).__init__(name=name)
        self._transition_model = transition_model

    def propose(self, state: State, inputs: tf.Tensor, observation: tf.Tensor, seed=None):
        """See base class"""
        proposed_particles = self._transition_model.sample(state, inputs, seed=seed)
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
        return self._transition_model.loglikelihood(state, proposed_state, inputs)
