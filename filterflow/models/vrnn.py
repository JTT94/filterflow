import attr
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.observation.base import ObservationModelBase
from filterflow.proposal.base import ProposalModelBase
from filterflow.smc import SMC
from filterflow.transition.base import TransitionModelBase

tfd = tfp.distributions


@attr.s
class VRNNState(State):
    ADDITIONAL_STATE_VARIABLES = ('rnn_state',)  # rnn_out and encoded no need to be resampled
    rnn_state = attr.ib(default=None)
    rnn_out = attr.ib(default=None)
    latent_encoded = attr.ib(default=None)


class NNNormalDistribution(tf.Module):
    """A Normal distribution with mean and var parametrised by NN"""

    def __init__(self,
                 size,
                 hidden_layer_sizes,
                 sigma_min=0.0,
                 raw_sigma_bias=0.25,
                 hidden_activation_fn=tf.nn.relu,
                 name="conditional_normal_distribution"):
        super(NNNormalDistribution, self).__init__(name=name)

        self.sigma_min = sigma_min
        self.raw_sigma_bias = raw_sigma_bias
        self.size = size
        self.fcnet = snt.nets.MLP(
            output_sizes=hidden_layer_sizes + [2 * size],
            activation=hidden_activation_fn,
            activate_final=False,
            name=name + "_fcnet")

    def get_params(self, tensor_list, **_kwargs):
        """Computes the parameters of a normal distribution based on the inputs."""
        inputs = tf.concat(tensor_list, axis=-1)
        outs = self.fcnet(inputs)
        mu, sigma = tf.split(outs, 2, axis=-1)
        sigma = tf.maximum(tf.nn.softplus(sigma + self.raw_sigma_bias), self.sigma_min)
        return mu, sigma

    def __call__(self, *args, **kwargs):
        """Creates a normal distribution conditioned on the inputs."""
        mu, sigma = self.get_params(args, **kwargs)
        return tfp.distributions.Normal(loc=mu, scale=sigma)


class NNBernoulliDistribution(tf.Module):
    """A Normal distribution with mean and var parametrised by NN"""

    def __init__(self,
                 size,
                 hidden_layer_sizes,
                 hidden_activation_fn=tf.nn.relu,
                 name="conditional_bernoulli_distribution"):
        super(NNBernoulliDistribution, self).__init__(name=name)

        self.size = size
        self.fcnet = snt.nets.MLP(
            output_sizes=hidden_layer_sizes + [size],
            activation=hidden_activation_fn,
            activate_final=False,
            name=name + "_fcnet")

    def get_logits(self, tensor_list, **_kwargs):
        """Computes the parameters of a normal distribution based on the inputs."""
        inputs = tf.concat(tensor_list, axis=-1)
        return self.fcnet(inputs)

    def __call__(self, *args, **kwargs):
        """Creates a normal distribution conditioned on the inputs."""
        logits = self.get_logits(args, **kwargs)
        return tfp.distributions.Bernoulli(logits=logits)


class VRNNTransitionModel(TransitionModelBase):
    def __init__(self,
                 rnn_hidden_size,
                 latent_size,
                 data_encoder,
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

    def run_rnn(self, state: VRNNState, inputs: tf.Tensor):
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

    def loglikelihood(self, prior_state: VRNNState, proposed_state: VRNNState, inputs: tf.Tensor):
        rnn_out, rnn_state, latent_encoded = self.run_rnn(prior_state, inputs)
        dist = self.transition(rnn_out)
        new_latent = proposed_state.particles
        return tf.reduce_sum(dist.log_prob(new_latent), axis=-1)

    def sample(self, state: VRNNState, inputs: tf.Tensor, seed=None):
        rnn_out, rnn_state, latent_encoded = self.run_rnn(state, inputs)
        dist = self.latent_dist(state, rnn_out)
        latent_state = dist.sample(seed=seed)

        return VRNNState(particles=latent_state,
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


class VRNNNormalObservationModel(ObservationModelBase):

    def __init__(self, latent_encoder, observation_size, name='VRNNObservationModel'):
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

    def __init__(self, latent_encoder, observation_size, name='VRNNObservationModel'):
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

    def sample(self, state: State):
        dist = self.observation_dist(state)
        return dist.sample()


class ObservationModelEnum:
    BERNOULLI = 0
    NORMAL = 1


def make_filter(latent_size, observation_size, rnn_hidden_size, latent_encoder_layers,
                latent_encoded_size, observation_model_name, resampling_method, resampling_criterion):
    data_encoder = None

    if observation_model_name == ObservationModelEnum.NORMAL:
        observation_model = VRNNNormalObservationModel(data_encoder, observation_size)
    elif observation_model_name == ObservationModelEnum.BERNOULLI:
        observation_model = VRNNBernoulliObservationModel(data_encoder, observation_size)
    else:
        raise ValueError(f'observation_model_name {observation_model_name} '
                         f'is not a valid value for ObservationModelEnum')

    latent_encoder = snt.nets.MLP(
        output_sizes=latent_encoder_layers + [latent_encoded_size],
        name="latent_encoder")

    transition_model = VRNNTransitionModel(rnn_hidden_size, latent_size, data_encoder, latent_encoder)
    proposal_model = VRNNProposalModel(transition_model)

    return SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)
