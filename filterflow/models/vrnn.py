import os
import sys
import attr

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

import sonnet as snt

from filterflow.smc import SMC
from filterflow.base import State

from filterflow.observation.base import ObservationModelBase, ObservationSampler
from filterflow.proposal.base import ProposalModelBase
from filterflow.transition.base import TransitionModelBase
from filterflow.resampling.criterion import NeffCriterion, ResamplingCriterionBase
from filterflow.resampling.standard import MultinomialResampler
from filterflow.resampling.differentiable import RegularisedTransform

from filterflow.resampling.base import ResamplerBase
from filterflow.utils import normalize
from filterflow.models import NNNormalDistribution, NNBernoulliDistribution

## Initial State

@attr.s
class VRNNState(State):
    ADDITIONAL_STATE_VARIABLES = ('rnn_state',)
    rnn_state = attr.ib(default=None)
    rnn_out = attr.ib(default=None)
    obs_likelihood = attr.ib(default=None)
    latent_encoded = attr.ib(default=None)


## Transition

class VRNNTransitionModel(TransitionModelBase):
    def __init__(self,
                 rnn_hidden_size,
                 latent_size,
                 latent_encoder,
                 name='NNTransitionModel'):
        super(VRNNTransitionModel, self).__init__(name=name)

        # mlp parametrised gaussian
        self.transition = NNNormalDistribution(size=latent_size,
                                               hidden_layer_sizes=[latent_size])
        # encoder for inputs
        self.latent_encoder = latent_encoder

        # lstm cell
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = tf.keras.layers.LSTMCell(rnn_hidden_size)

    def run_rnn(self, state: State, inputs: tf.Tensor):
        tiled_inputs = tf.tile(inputs, [state.batch_size, state.n_particles, 1])
        # process latent state
        latent_state = state.particles

        # encode and reshape latent state
        latent_encoded = self.latent_encoder(latent_state)

        B, N, D = latent_encoded.shape
        # process rnn_state
        rnn_state = tf.reshape(state.rnn_state, [B, N, self.rnn_hidden_size * 2])

        rnn_state = tf.split(rnn_state, 2, axis=-1)

        # run rnn
        rnn_inputs = tf.concat([tiled_inputs, latent_encoded], axis=-1)
        rnn_inputs_reshaped = tf.reshape(rnn_inputs, (B * N, -1))
        rnn_state_reshaped = [tf.reshape(elem, (B * N, -1)) for elem in rnn_state]
        rnn_out, rnn_state = self.rnn(rnn_inputs_reshaped, rnn_state_reshaped)

        rnn_state = tf.concat(rnn_state, axis=-1)
        rnn_state = tf.reshape(rnn_state, [state.batch_size, state.n_particles, self.rnn_hidden_size * 2])
        rnn_out = tf.reshape(rnn_out, [state.batch_size, state.n_particles, self.rnn_hidden_size])
        return rnn_out, rnn_state, latent_encoded

    def latent_dist(self, state, rnn_out):
        dist = self.transition(rnn_out)
        return dist

    def loglikelihood(self, prior_state: State, proposed_state: State, inputs: tf.Tensor):
        rnn_out, rnn_state, latent_encoded = self.run_rnn(prior_state, inputs)
        dist = self.transition(rnn_out)
        new_latent = proposed_state.particles
        return tf.reduce_sum(dist.log_prob(new_latent), axis=-1)

    def sample(self, state: State, inputs: tf.Tensor, seed=None):
        rnn_out, rnn_state, latent_encoded = self.run_rnn(state, inputs)
        dist = self.latent_dist(state, rnn_out)
        latent_state = dist.sample(seed=seed)

        return attr.evolve(state, particles=latent_state,
                           log_weights=state.log_weights,
                           weights=state.weights,
                           log_likelihoods=state.log_likelihoods,
                           rnn_state=rnn_state,
                           rnn_out=rnn_out,
                           latent_encoded=latent_encoded)


class VRNNProposalModel(ProposalModelBase):
    def __init__(self,
                 transition_model,
                 name='VRNNProposalModel'):
        super(VRNNProposalModel, self).__init__(name=name)
        self._transiton_model = transition_model

    def loglikelihood(self, proposed_state: State, state: State, inputs: tf.Tensor, observation: tf.Tensor):
        rnn_out, rnn_state, latent_encoded = self._transiton_model.run_rnn(state, inputs)
        dist = self._transiton_model.latent_dist(state, rnn_out)
        new_latent = proposed_state.particles
        return tf.reduce_sum(dist.log_prob(new_latent), axis=-1)

    def propose(self, state: State, inputs: tf.Tensor, observation: tf.Tensor, seed=None):
        return self._transiton_model.sample(state, inputs, seed)


class TESTVRNNTransitionModel(VRNNTransitionModel):
    def __init__(self,
                 rnn_hidden_size,
                 latent_encoder,
                 latent_size,
                 name='NNTransitionModel'):
        super(VRNNTransitionModel, self).__init__(name=name)

        # mlp parametrised gaussian
        self.transition = lambda x: tfp.distributions.Normal(loc=0., scale=1.)
        # encoder for inputs
        self.latent_encoder = latent_encoder

        # lstm cell
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = tf.keras.layers.LSTMCell(rnn_hidden_size)

    def sample(self, state: State, inputs: tf.Tensor, seed=None):
        rnn_out, rnn_state, latent_encoded = self.run_rnn(state, inputs)
        dist = self.latent_dist(state, rnn_out)
        latent_state = dist.sample([state.batch_size, state.n_particles, state.dimension], seed=seed)

        return attr.evolve(state, particles=latent_state,
                           log_weights=state.log_weights,
                           weights=state.weights,
                           log_likelihoods=state.log_likelihoods,
                           rnn_state=rnn_state,
                           rnn_out=rnn_out,
                           latent_encoded=latent_encoded)


class TESTVRNNProposalModel(VRNNProposalModel):
    def __init__(self,
                 rnn_hidden_size,
                 latent_encoder,
                 latent_size,
                 name='VRNNProposalModel'):
        super(VRNNProposalModel, self).__init__(rnn_hidden_size, latent_encoder, latent_size, name)



## Observation Model
class VRNNNormalObservationModel(ObservationModelBase):

    def __init__(self, observation_size, name='VRNNObservationModel'):
        super(VRNNNormalObservationModel, self).__init__(name=name)
        # mlp parametrised gaussian
        self.emission = NNNormalDistribution(size=observation_size,
                                             hidden_layer_sizes=[observation_size])

    def observation_dist(self, state: VRNNState):
        latent_state = state.particles
        latent_encoded = state.latent_encoded
        rnn_out = state.rnn_out
        dist = self.emission(latent_state, rnn_out)
        return dist

    def loglikelihood(self, state: State, observation: tf.Tensor):
        dist = self.observation_dist(state)
        return tf.reduce_sum(dist.log_prob(observation), axis=-1)


class VRNNBernoulliObservationModel(ObservationModelBase):

    def __init__(self, observation_size, name='VRNNObservationModel'):
        super(VRNNBernoulliObservationModel, self).__init__(name=name)
        # mlp parametrised gaussian
        self.emission = NNBernoulliDistribution(size=observation_size,
                                                hidden_layer_sizes=[observation_size])

    def observation_dist(self, state: State):
        latent_state = state.particles
        latent_encoded = state.latent_encoded
        rnn_out = state.rnn_out
        dist = self.emission(latent_state, rnn_out)
        return dist

    def loglikelihood(self, state: State, observation: tf.Tensor):
        dist = self.observation_dist(state)
        return tf.reduce_sum(dist.log_prob(observation), axis=-1)

    def sample(self, state: State, seed=None):
        dist = self.observation_dist(state)
        return dist.sample(seed=seed)



class VRNNSMC(SMC):
    def __init__(self, observation_model: ObservationModelBase, transition_model: TransitionModelBase,
                 proposal_model: ProposalModelBase, resampling_criterion: ResamplingCriterionBase,
                 resampling_method: ResamplerBase, name='VRNNSMC'):
        super(VRNNSMC, self).__init__(observation_model,
                                      transition_model,
                                      proposal_model,
                                      resampling_criterion, resampling_method, name=name)

    @tf.function
    def propose_and_weight(self, state: State, observation: tf.Tensor,
                           inputs: tf.Tensor, seed=None):
        """
        :param state: State
            current state of the filter
        :param observation: tf.Tensor
            observation to compare the state against
        :param inputs: tf.Tensor
            inputs for the observation_model
        :return: Updated weights
        """
        proposed_state = self._proposal_model.propose(state, inputs, observation, seed=seed)

        log_weights = self._transition_model.loglikelihood(state, proposed_state, inputs)
        obs_likelihood = self._observation_model.loglikelihood(proposed_state, observation)
        log_weights = log_weights + obs_likelihood
        log_weights = log_weights - self._proposal_model.loglikelihood(proposed_state, state, inputs, observation)
        log_weights = log_weights + state.log_weights

        log_likelihood_increment = tf.math.reduce_logsumexp(log_weights, 1)
        log_likelihoods = state.log_likelihoods + log_likelihood_increment
        normalized_log_weights = normalize(log_weights, 1, state.n_particles, True)

        return attr.evolve(proposed_state,
                           obs_likelihood=tf.reduce_sum(obs_likelihood, -1),
                           weights=tf.math.exp(normalized_log_weights),
                           log_weights=normalized_log_weights,
                           log_likelihoods=log_likelihoods)


def make_filter(latent_size, observation_size, rnn_hidden_size, latent_encoder_layers,
                latent_encoded_size, resampling_method, resampling_criterion):
    observation_model = VRNNBernoulliObservationModel(observation_size)

    latent_encoder = snt.nets.MLP(
        output_sizes=latent_encoder_layers + [latent_encoded_size],
        name="latent_encoder")

    transition_model = VRNNTransitionModel(rnn_hidden_size, latent_size, latent_encoder)
    proposal_model = VRNNProposalModel(transition_model)

    return SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)


