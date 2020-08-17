import os
import sys

# add to path
sys.path.append("../")
from absl import flags, app
import attr

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

tf.config.set_visible_devices([], 'GPU')

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

## Load Data

import pickle

from scipy.sparse import coo_matrix


def sparse_pianoroll_to_dense(pianoroll, min_note, num_notes):
    """Converts a sparse pianoroll to a dense numpy array.
    Given a sparse pianoroll, converts it to a dense numpy array of shape
    [num_timesteps, num_notes] where entry i,j is 1.0 if note j is active on
    timestep i and 0.0 otherwise.
    Args:
    pianoroll: A sparse pianoroll object, a list of tuples where the i'th tuple
      contains the indices of the notes active at timestep i.
    min_note: The minimum note in the pianoroll, subtracted from all notes so
      that the minimum note becomes 0.
    num_notes: The number of possible different note indices, determines the
      second dimension of the resulting dense array.
    Returns:
    dense_pianoroll: A [num_timesteps, num_notes] numpy array of floats.
    num_timesteps: A python int, the number of timesteps in the pianoroll.
    """
    num_timesteps = len(pianoroll)
    inds = []
    for time, chord in enumerate(pianoroll):
        # Re-index the notes to start from min_note.
        inds.extend((time, note - min_note) for note in chord)
        shape = [num_timesteps, num_notes]
    values = [1.] * len(inds)
    sparse_pianoroll = coo_matrix(
        (values, ([x[0] for x in inds], [x[1] for x in inds])),
        shape=shape)
    return sparse_pianoroll.toarray(), num_timesteps


def create_pianoroll_dataset(path,
                             split,
                             batch_size,
                             num_parallel_calls=4,
                             shuffle=False,
                             repeat=False,
                             min_note=21,
                             max_note=108):
    """Creates a pianoroll dataset.
    Args:
    path: The path of a pickle file containing the dataset to load.
    split: The split to use, can be train, test, or valid.
    batch_size: The batch size. If repeat is False then it is not guaranteed
      that the true batch size will match for all batches since batch_size
      may not necessarily evenly divide the number of elements.
    num_parallel_calls: The number of threads to use for parallel processing of
      the data.
    shuffle: If true, shuffles the order of the dataset.
    repeat: If true, repeats the dataset endlessly.
    min_note: The minimum note number of the dataset. For all pianoroll datasets
      the minimum note is number 21, and changing this affects the dimension of
      the data. This is useful mostly for testing.
    max_note: The maximum note number of the dataset. For all pianoroll datasets
      the maximum note is number 108, and changing this affects the dimension of
      the data. This is useful mostly for testing.
    Returns:
    inputs: A batch of input sequences represented as a dense Tensor of shape
      [time, batch_size, data_dimension]. The sequences in inputs are the
      sequences in targets shifted one timestep into the future, padded with
      zeros. This tensor is mean-centered, with the mean taken from the pickle
      file key 'train_mean'.
    targets: A batch of target sequences represented as a dense Tensor of
      shape [time, batch_size, data_dimension].
    lens: An int Tensor of shape [batch_size] representing the lengths of each
      sequence in the batch.
    mean: A float Tensor of shape [data_dimension] containing the mean loaded
      from the pickle file.
    """
    # Load the data from disk.
    num_notes = max_note - min_note + 1
    with tf.io.gfile.GFile(path, "rb") as f:
        raw_data = pickle.load(f)
    pianorolls = raw_data[split]
    mean = raw_data["train_mean"]
    num_examples = len(pianorolls)

    def pianoroll_generator():
        for sparse_pianoroll in pianorolls:
            yield sparse_pianoroll_to_dense(sparse_pianoroll, min_note, num_notes)

    dataset = tf.data.Dataset.from_generator(
        pianoroll_generator,
        output_types=(tf.float64, tf.int64),
        output_shapes=([None, num_notes], []))

    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(num_examples)

    # Batch sequences togther, padding them to a common length in time.
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=([None, num_notes], []))

    def process_pianoroll_batch(data, lengths):
        """Create mean-centered and time-major next-step prediction Tensors."""
        data = tf.cast(tf.transpose(data, perm=[1, 0, 2]), float)
        lengths = tf.cast(lengths, tf.int32)
        targets = data
        # Mean center the inputs.
        inputs = data - tf.constant(mean, dtype=tf.float32,
                                    shape=[1, 1, mean.shape[0]])
        # Shift the inputs one step forward in time. Also remove the last timestep
        # so that targets and inputs are the same length.
        inputs = tf.pad(inputs, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
        # Mask out unused timesteps.
        inputs *= tf.expand_dims(tf.transpose(
            tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
        return inputs, targets, lengths

    dataset = dataset.map(process_pianoroll_batch,
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(num_examples)

    itr = tf.compat.v1.data.make_one_shot_iterator(dataset)
    inputs, targets, lengths = itr.get_next()
    return inputs, targets, lengths, tf.constant(mean, dtype=tf.float32)


def fn_identifier(initial_lr, decay, steps, method, data_name):
    return "method_{3}__lr0_{0}__decay_{1}__steps_{2}_data_{4}".format(initial_lr, decay, steps, method, data_name)


def pickle_obj(obj, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_obj(file_path):
    with open(file_path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


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

    def get_params(self, tensor_list, **unused_kwargs):
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

    def get_logits(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a normal distribution based on the inputs."""
        inputs = tf.concat(tensor_list, axis=-1)
        return self.fcnet(inputs)

    def __call__(self, *args, **kwargs):
        """Creates a normal distribution conditioned on the inputs."""
        logits = self.get_logits(args, **kwargs)
        return tfp.distributions.Bernoulli(logits=logits)


## Transition


class VRNNTransitionModel(TransitionModelBase):
    def __init__(self,
                 rnn_hidden_size,
                 latent_encoder,
                 latent_size,
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


class VRNNProposalModel(VRNNTransitionModel):
    def __init__(self,
                 rnn_hidden_size,
                 latent_encoder,
                 latent_size,
                 name='VRNNProposalModel'):
        super(VRNNProposalModel, self).__init__(rnn_hidden_size, latent_encoder, latent_size, name)

    def loglikelihood(self, proposed_state: State, state: State, inputs: tf.Tensor, observation: tf.Tensor):
        rnn_out, rnn_state, latent_encoded = self.run_rnn(state, inputs)
        dist = self.latent_dist(state, rnn_out)
        new_latent = proposed_state.particles
        return tf.reduce_sum(dist.log_prob(new_latent), axis=-1)

    def propose(self, state: State, inputs: tf.Tensor, observation: tf.Tensor, seed=None):
        return self.sample(state, inputs, seed=seed)


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

class VRNNBernoulliObservationModel(ObservationSampler):

    def __init__(self, latent_encoder, observation_size, name='VRNNObservationModel'):
        super(VRNNBernoulliObservationModel, self).__init__(name=name)
        # mlp parametrised gaussian
        self.emission = NNBernoulliDistribution(size=observation_size,
                                                hidden_layer_sizes=[observation_size])

    def observation_dist(self, state: State):
        latent_state = state.particles
        latent_encoded = state.latent_encoded
        rnn_out = state.rnn_out
        dist = self.emission(latent_encoded, rnn_out)
        return dist

    def loglikelihood(self, state: State, observation: tf.Tensor):
        dist = self.observation_dist(state)
        return tf.reduce_sum(dist.log_prob(observation), axis=-1)

    def sample(self, state: State, seed=None):
        dist = self.observation_dist(state)
        return dist.sample(seed=seed)


## Initial State

@attr.s
class VRNNState(State):
    ADDITIONAL_STATE_VARIABLES = ('rnn_state',)
    rnn_state = attr.ib(default=None)
    rnn_out = attr.ib(default=None)
    obs_likelihood = attr.ib(default=None)
    latent_encoded = attr.ib(default=None)


def make_optimizer(initial_learning_rate=0.01, decay_steps=100, decay_rate=0.75, staircase=True):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase)

    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
    return optimizer


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


# -----------------------------------------------------------

def main(run_method,
         latent_size=10,
         latent_encoded_size=32,
         batch_size=1,
         n_particles=25,
         epsilon=0.5,
         scaling=0.9,
         neff=0.9,
         max_iter=1000,
         additional_variables_are_state=False,
         convergence_threshold=1e-3,
         n_iter=100,
         initial_lr=0.01,
         decay=0.5,
         steps=100,
         warmup=100,
         data_seed=0,
         filter_seed=1,
         fixed_seed=False,
         out_dir='./',
         data_fp='../data/data/piano_data/jsb.pkl'):
    inputs_tensor, targets_tensor, lens, mean = create_pianoroll_dataset(data_fp, split='train', batch_size=1)

    T = targets_tensor.shape.as_list()[0]
    observation_size = targets_tensor.shape.as_list()[-1]

    encoded_data_size = latent_size
    rnn_hidden_size = latent_size // 2

    latent_encoder_layers = [32]

    latent_encoder = snt.nets.MLP(
        output_sizes=latent_encoder_layers + [latent_encoded_size],
        name="latent_encoder")

    # store observations

    dimension = latent_size
    inputs_tensor = tf.expand_dims(inputs_tensor, 1)
    targets_tensor = tf.expand_dims(targets_tensor, 1)

    obs_data = tf.data.Dataset.from_tensor_slices(targets_tensor)
    inputs_data = tf.data.Dataset.from_tensor_slices(inputs_tensor)
    transition_model = VRNNTransitionModel(rnn_hidden_size, latent_encoder, latent_size)
    observation_model = VRNNBernoulliObservationModel(latent_encoder, observation_size)
    proposal_model = VRNNProposalModel(rnn_hidden_size, latent_encoder, latent_size)

    test_transition_model = TESTVRNNTransitionModel(rnn_hidden_size, latent_encoder, latent_size)
    test_proposal_model = TESTVRNNProposalModel(rnn_hidden_size, latent_encoder, latent_size)

    # initial state
    tf.random.set_seed(data_seed)
    normal_dist = tfp.distributions.Normal(0., 1.)
    initial_latent_state = tf.zeros([batch_size, n_particles, dimension])
    initial_latent_state = tf.cast(initial_latent_state, dtype=float)
    latent_encoded = transition_model.latent_encoder(initial_latent_state)

    # initial rnn_state
    initial_rnn_state = [normal_dist.sample([batch_size, n_particles, rnn_hidden_size], seed=data_seed)] * 2
    initial_rnn_state = tf.concat(initial_rnn_state, axis=-1)

    # rnn_out
    initial_rnn_out = tf.zeros([batch_size, n_particles, rnn_hidden_size])

    initial_weights = tf.ones((batch_size, n_particles), dtype=float) / tf.cast(n_particles, float)
    log_likelihoods = tf.zeros(batch_size, dtype=float)
    init_state = VRNNState(particles=initial_latent_state,
                           log_weights=tf.math.log(initial_weights),
                           weights=initial_weights,
                           obs_likelihood=log_likelihoods,
                           log_likelihoods=log_likelihoods,
                           rnn_state=initial_rnn_state,
                           rnn_out=initial_rnn_out,
                           latent_encoded=latent_encoded)

    # record loss
    LARGE_B = 50
    N = 25

    # initial state
    large_initial_latent_state = tf.zeros([LARGE_B, N, dimension])
    large_initial_latent_state = tf.cast(large_initial_latent_state, dtype=float)
    large_latent_encoded = transition_model.latent_encoder(large_initial_latent_state)

    # initial rnn_state
    large_initial_rnn_state = [normal_dist.sample([LARGE_B, N, rnn_hidden_size])] * 2
    large_initial_rnn_state = tf.concat(large_initial_rnn_state, axis=-1)

    # rnn_out
    large_initial_rnn_out = tf.zeros([LARGE_B, N, rnn_hidden_size])
    obs_likelihood = tf.zeros(LARGE_B, dtype=float)
    large_init_state = VRNNState(particles=large_initial_latent_state,
                                 obs_likelihood=obs_likelihood,
                                 rnn_state=large_initial_rnn_state,
                                 rnn_out=large_initial_rnn_out,
                                 latent_encoded=large_latent_encoded)
    ## Check variables

    # snt networks initiated on first call
    t_samp = transition_model.sample(init_state, inputs_tensor[0], seed=data_seed)
    obs_samp = observation_model.sample(init_state, seed=data_seed)

    # for var in transition_model.variables:
    #    print(var.name)

    # for var in observation_model.variables:
    #    print(var.name)

    ## Particle Filter

    trainable_variables = transition_model.variables + observation_model.variables
    init_values = [v.value() for v in trainable_variables]

    resampling_criterion = NeffCriterion(tf.constant(neff), tf.constant(True))
    # resampling_criterion = AlwaysResample()
    resampling_method = MultinomialResampler()

    epsilon = tf.constant(epsilon)
    scaling = tf.constant(scaling)

    regularized = RegularisedTransform(epsilon,
                                       scaling=scaling,
                                       max_iter=max_iter,
                                       convergence_threshold=convergence_threshold,
                                       additional_variables_are_state=additional_variables_are_state)

    multinomial_smc = VRNNSMC(observation_model, transition_model, proposal_model, resampling_criterion, MultinomialResampler())
    regularized_smc = VRNNSMC(observation_model, transition_model, proposal_model, resampling_criterion, regularized)
    test_reg = VRNNSMC(observation_model, test_transition_model, test_proposal_model, resampling_criterion, regularized)
    test_mul = VRNNSMC(observation_model, test_transition_model, test_proposal_model, resampling_criterion, MultinomialResampler())

    def run_smc(smc, optimizer, n_iter, seed=filter_seed):
        # print(optimizer.weights)# check
        @tf.function
        def smc_routine(smc, state, use_correction_term=False, seed=seed):
            final_state = smc(state, obs_data, n_observations=T, inputs_series=inputs_data, return_final=True,
                              seed=seed)
            res = tf.reduce_mean(final_state.log_likelihoods)
            obs_likelihood = tf.reduce_mean(final_state.obs_likelihood)
            ess = final_state.ess
            if use_correction_term:
                return res, tf.reduce_mean(final_state.resampling_correction)
            return res, ess, tf.constant(0.), obs_likelihood

        @tf.function
        def run_one_step(smc, use_correction_term, init_state, seed=seed):
            with tf.GradientTape() as tape:
                tape.watch(trainable_variables)
                real_ll, ess, correction, obs_likelihood = smc_routine(smc, init_state, use_correction_term, seed)
                loss = -(real_ll + correction)
            grads_loss = tape.gradient(loss, trainable_variables)
            return real_ll, grads_loss, ess, obs_likelihood

        @tf.function
        def train_one_step(smc, use_correction_term, seed=seed):
            real_ll, grads_loss, ess, obs_likelihood = run_one_step(smc, use_correction_term, init_state, seed)
            capped_gvs = [tf.clip_by_value(grad, -500., 500.) for grad in grads_loss]
            optimizer.apply_gradients(zip(capped_gvs, trainable_variables))
            return -real_ll, capped_gvs, ess, obs_likelihood

        @tf.function
        def train_niter(smc, num_steps=100, use_correction_term=False, reset=True, seed=seed, fixed_seed=fixed_seed):
            if reset:
                reset_operations = [v.assign(init) for v, init in zip(trainable_variables, init_values)]
            else:
                reset_operations = []
            obs_lik_tensor_array = tf.TensorArray(dtype=tf.float32, size=num_steps, dynamic_size=False,
                                                  element_shape=[])
            multi_loss_tensor_array = tf.TensorArray(dtype=tf.float32, size=num_steps, dynamic_size=False,
                                                     element_shape=[])
            test_reg_tensor_array = tf.TensorArray(dtype=tf.float32, size=num_steps, dynamic_size=False,
                                                     element_shape=[])          
            test_mul_tensor_array = tf.TensorArray(dtype=tf.float32, size=num_steps, dynamic_size=False,
                                                     element_shape=[])                                                                                 
            loss_tensor_array = tf.TensorArray(dtype=tf.float32, size=num_steps, dynamic_size=False, element_shape=[])
            ess_tensor_array = tf.TensorArray(dtype=tf.float32, size=num_steps, dynamic_size=False, element_shape=[])
            grad_tensor_array = tf.TensorArray(dtype=tf.float32, size=num_steps, dynamic_size=False, element_shape=[])
            time_tensor_array = tf.TensorArray(dtype=tf.float64, size=num_steps, dynamic_size=False, element_shape=[])
            with tf.control_dependencies(reset_operations):
                toc = tf.constant(0., dtype=tf.float64)
                tic = tf.timestamp()
                for step in tf.range(1, num_steps + 1):
                    if fixed_seed:
                        seed = seed
                    else:
                        seed = step
                    tic_loss = tf.timestamp()
                    with tf.control_dependencies([tic_loss]):
                        loss, grads, ess_run, obs_likelihood = train_one_step(smc, use_correction_term, seed)
                    with tf.control_dependencies([loss]):
                        toc_loss = tf.timestamp()
                        multi_loss_state = multinomial_smc(large_init_state, obs_data,
                                                           n_observations=T, inputs_series=inputs_data,
                                                           return_final=True, seed=seed)

                        test_reg_state = test_reg(large_init_state, obs_data,
                                                           n_observations=T, inputs_series=inputs_data,
                                                           return_final=True, seed=seed)

                        test_mul_state = test_mul(large_init_state, obs_data,
                                                           n_observations=T, inputs_series=inputs_data,
                                                           return_final=True, seed=seed)                                   
                        test_reg_loss = -tf.reduce_mean(test_reg_state.log_likelihoods)
                        test_mul_loss = -tf.reduce_mean(test_mul_state.log_likelihoods)
                        multi_loss = -tf.reduce_mean(multi_loss_state.log_likelihoods)
                        ess = multi_loss_state.ess
                    toc += toc_loss - tic_loss

                    max_grad = tf.reduce_max([tf.reduce_max(tf.abs(grad)) for grad in grads])

                    print_step = num_steps // 10
                    if step % print_step == 0:
                        tf.print('Step', step, '/', num_steps,
                                 ', obs_likelihood = ', obs_likelihood,
                                 ', loss = ', loss,
                                 ', test_reg = ', test_reg_loss,
                                 ', test_mul = ', test_mul_loss,
                                 ', multi_loss= ', multi_loss,
                                 ': ms per step= ', 1000. * toc / tf.cast(step, tf.float64),
                                 end='\r')

                    test_reg_tensor_array = test_reg_tensor_array.write(step - 1, test_reg_loss)
                    test_mul_tensor_array = test_mul_tensor_array.write(step - 1, test_mul_loss)
                    obs_lik_tensor_array = obs_lik_tensor_array.write(step - 1, obs_likelihood)
                    multi_loss_tensor_array = multi_loss_tensor_array.write(step - 1, multi_loss)
                    ess_tensor_array = ess_tensor_array.write(step - 1, ess[0])
                    loss_tensor_array = loss_tensor_array.write(step - 1, loss)
                    grad_tensor_array = grad_tensor_array.write(step - 1, max_grad)
                    time_tensor_array = time_tensor_array.write(step - 1, toc)

            return (loss_tensor_array.stack(), grad_tensor_array.stack(),
                    time_tensor_array.stack(), ess_tensor_array.stack(),
                    multi_loss_tensor_array.stack(), obs_lik_tensor_array.stack(), 
                    test_reg_tensor_array.stack(), test_mul_tensor_array.stack())

        return train_niter(smc, tf.constant(n_iter))

    def run_block(smc, method, n_iter, initial_lr, decay, steps, out_dir, col='blue', warnup=100, force=False,
                  data_name=None):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        optimizer = make_optimizer(initial_learning_rate=initial_lr,
                                   decay_steps=steps, decay_rate=decay, staircase=True)
        key = fn_identifier(initial_lr, decay, steps, method, data_name)
        filename = "vrnn_loss_{0}.pkl".format(key)
        filepath = os.path.join(out_dir, filename)

        print("\n {0}".format(method))

        print(key)

        (loss_array,
         grad_array,
         time_array,
         ess_array,
         multi_loss_array,
         obs_lik_array,
         test_reg_array, test_mul_array) = run_smc(smc, optimizer, n_iter, seed=filter_seed)

        obs_lik_array = obs_lik_array.numpy()
        loss_array = loss_array.numpy()
        grad_array = grad_array.numpy()
        time_array = time_array.numpy()
        ess_array = ess_array.numpy()
        test_reg_array= test_reg_array.numpy()
        test_mul_array= test_mul_array.numpy()
        multi_loss_array = multi_loss_array.numpy()

        pickle_obj(loss_array, os.path.join(out_dir, filename))

        filename_test_loss = "vrnn_reg_tloss_{0}.pkl".format(key)
        pickle_obj(test_reg_array, os.path.join(out_dir, filename_test_loss))

        filename_test_loss = "vrnn_mul_tloss_{0}.pkl".format(key)
        pickle_obj(test_mul_array, os.path.join(out_dir, filename_test_loss))

        filename_olik = "vrnn_olik_{0}.pkl".format(key)
        pickle_obj(obs_lik_array, os.path.join(out_dir, filename_olik))

        filename_mloss = "vrnn_mloss_{0}.pkl".format(key)
        pickle_obj(multi_loss_array, os.path.join(out_dir, filename_mloss))

        filename_ess = "vrnn_ess_{0}.pkl".format(key)
        pickle_obj(ess_array, os.path.join(out_dir, filename_ess))
        filename_grad = "vrnn_grad_{0}.pkl".format(key)
        pickle_obj(grad_array, os.path.join(out_dir, filename_grad))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ess_array, color=col)
        fig.savefig(os.path.join(out_dir, 'vrnn_ess_{0}.png'.format(key)))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(grad_array, color=col)
        fig.savefig(os.path.join(out_dir, 'vrnn_grad_{0}.png'.format(key)))
        plt.close()

        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot(loss_array[warmup:], color=col)
        # fig.savefig(os.path.join(out_dir, 'vrnn_loss_{0}.png'.format(key)))
        # plt.close()

        return multi_loss_array

    print(run_method)
    data_name = os.path.splitext(os.path.basename(data_fp))[0]

    if run_method == 'mult':
        multi_array = run_block(multinomial_smc, 'mult', n_iter, initial_lr, decay, steps, out_dir, col='blue',
                                data_name=data_name)

    if run_method == 'reg':
        print(resampling_method)
        reg_array = run_block(regularized_smc, 'reg', n_iter, initial_lr, decay, steps, out_dir, col='green',
                              data_name=data_name)
    # both_key = fn_identifier(initial_lr, decay, steps, 'both')

    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(multi_array[warmup:], color='blue')
    # ax.plot(reg_array[warmup:], color='green')
    # fig.savefig(os.path.join(out_dir, 'vrnn_loss_{0}.png'.format(both_key)))
    # plt.close()
    # print('vrnn_loss_{0}.png'.format(both_key))
    # print("\n\n")


# define flags

FLAGS = flags.FLAGS

flags.DEFINE_float('epsilon', 0.5, 'epsilon')
flags.DEFINE_float('resampling_neff', 0.5, 'resampling_neff')
flags.DEFINE_float('scaling', 0.9, 'scaling')
flags.DEFINE_float('convergence_threshold', 1e-3, 'convergence_threshold')
flags.DEFINE_float('initial_lr', 0.01, 'initial_lr')
flags.DEFINE_float('decay', 0.75, 'decay')
flags.DEFINE_integer('decay_steps', 100, 'decay_steps')
flags.DEFINE_boolean('additional_variables_are_state', True, 'Use the RNN state to compute the transport matrix')
flags.DEFINE_integer('n_iter', 500, 'n_iter', lower_bound=1)
flags.DEFINE_integer('latent_size', 10, 'latent_size')
flags.DEFINE_integer('latent_encoded_size', 32, 'latent_encoded_size')
flags.DEFINE_integer('batch_size', 1, 'batch_size', lower_bound=1)
flags.DEFINE_integer('n_particles', 25, 'n_particles', lower_bound=4)
flags.DEFINE_integer('max_iter', 500, 'max_iter', lower_bound=1)
flags.DEFINE_integer('filter_seed', 42, 'filter_seed')
flags.DEFINE_integer('data_seed', 0, 'data_seed')
flags.DEFINE_string('out_dir', './', 'out_dir')
flags.DEFINE_string('resampling_method', 'reg', 'resampling method')
flags.DEFINE_boolean('fixed_filter_seed', False, 'fixed_filter_seed')
flags.DEFINE_string('data_fp', '../data/data/piano_data/jsb.pkl', 'data_fp')


def flag_main(argv):
    print('data_fp: {0}'.format(FLAGS.data_fp))
    print('resampling_method: {0}'.format(FLAGS.resampling_method))
    print('epsilon: {0}'.format(FLAGS.epsilon))
    print('resampling_neff: {0}'.format(FLAGS.resampling_neff))
    print('convergence_threshold: {0}'.format(FLAGS.convergence_threshold))
    print('additional_variables_are_state: {0}'.format(FLAGS.additional_variables_are_state))
    print('n_particles: {0}'.format(FLAGS.n_particles))
    print('scaling: {0}'.format(FLAGS.scaling))
    print('fixed_filter_seed: {0}'.format(FLAGS.fixed_filter_seed))
    print('filter_seed: {0}'.format(FLAGS.filter_seed))
    print('data_seed: {0}'.format(FLAGS.data_seed))
    print('max_iter: {0}'.format(FLAGS.max_iter))
    print('out_dir: {0}'.format(FLAGS.out_dir))
    print('initial_lr: {0}'.format(FLAGS.initial_lr))
    print('decay: {0}'.format(FLAGS.decay))
    print('decay_steps: {0}'.format(FLAGS.decay_steps))
    print('latent_size: {0}'.format(FLAGS.latent_size))
    print('latent_encoded_size: {0}'.format(FLAGS.latent_encoded_size))
    print('n_iter: {0}'.format(FLAGS.n_iter))

    main(run_method=FLAGS.resampling_method,
         latent_size=FLAGS.latent_size,
         latent_encoded_size=FLAGS.latent_encoded_size,
         batch_size=FLAGS.batch_size,
         n_particles=FLAGS.n_particles,
         epsilon=FLAGS.epsilon,
         scaling=FLAGS.scaling,
         neff=FLAGS.resampling_neff,
         max_iter=FLAGS.max_iter,
         additional_variables_are_state=FLAGS.additional_variables_are_state,
         convergence_threshold=FLAGS.convergence_threshold,
         n_iter=FLAGS.n_iter,
         initial_lr=FLAGS.initial_lr,
         decay=FLAGS.decay,
         steps=FLAGS.decay_steps,
         data_seed=FLAGS.data_seed,
         filter_seed=FLAGS.filter_seed,
         fixed_seed=FLAGS.fixed_filter_seed,
         out_dir=FLAGS.out_dir,
         data_fp=FLAGS.data_fp)


if __name__ == "__main__":
    app.run(flag_main)
