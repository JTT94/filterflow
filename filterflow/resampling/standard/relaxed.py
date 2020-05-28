# import tensorflow as tf
# import tensorflow_probability as tfp
#
# from filterflow.base import State
# from filterflow.resampling.base import ResamplerBase
#
#
# @tf.function
# def _blend_tensor(blending_weights, tensor, num_particles, batch_size):
#     """Blend tensor according to the weights.
#     The first dimension of tensor is actually a 2d index compacted to a 1d
#     index and similarly for blended_tensor. So if we index these Tensors
#     by [(i, j), k], then
#       blended_tensor[(i, j), k] =
#         sum_l tensor[(l, j), :] * blending_weights[i, j, l].
#     Args:
#       blending_weights: [num_particles, batch_size, num_particles] weights where
#         the indices represent [sample index, batch index, blending weight index].
#       tensor: [num_particles * batch_size, state_dim] Tensor to be blended.
#       num_particles: The number of particles/samples.
#       batch_size: The batch size.
#     Returns:
#       blended_tensor: [num_particles*batch_size, state_dim] blended Tensor.
#     """
#     # tensor is currently [num_particles * batch_size, state_dim], so we reshape
#     # it to [num_particles, batch_size, state_dim]. Then, transpose it to
#     # [batch_size, state_size, num_particles].
#     tensor = tf.transpose(
#         tf.reshape(tensor, [num_particles, batch_size, -1]), perm=[1, 2, 0])
#     blending_weights = tf.transpose(blending_weights, perm=[1, 2, 0])
#     # blendeding_weights is [batch index, blending weight index, sample index].
#     # Multiplying these gives a matrix of size [batch_size, state_size,
#     # num_particles].
#     tensor = tf.einsum('', tensor, blending_weights)
#     # transpose the tensor to be [num_particles, batch_size, state_size]
#     # and then reshape it to match the original format.
#     tensor = tf.reshape(tf.transpose(tensor, perm=[2, 0, 1]),
#                         [num_particles * batch_size, -1])
#     return tensor
#
#
# class RelaxedResampler(ResamplerBase):
#     """
#     This is an adaptation of resampling in FIVO
#     TODO: put a nice docstring here
#     """
#
#     DIFFERENTIABLE = False
#
#     def __init__(self, temperature, on_log=True, name='RelaxedResampler'):
#         super(RelaxedResampler, self).__init__(name)
#         self._temperature = temperature
#         self._on_log = on_log
#
#     def apply(self, state: State, flags: tf.Tensor):
#         """ Resampling method
#
#         :param state State
#             Particle filter state
#         :param flags: tf.Tensor
#             Flags for resampling
#         :return: resampled state
#         :rtype: State
#         """
#         batch_size = state.batch_size
#         n_particles = state.n_particles
#         # TODO: The real batch_size is the sum of flags. We shouldn't do more operations than we need...
#
#         if self._on_log:
#             log_weights = state.log_weights
#             resampling_dist = tfp.distributions.RelaxedOneHotCategorical(
#                 self._temperature,
#                 logits=log_weights)
#         else:
#             weights = state.weights
#             resampling_dist = tfp.distributions.RelaxedOneHotCategorical(
#                 self._temperature,
#                 probs=weights)
#
#         indices = resampling_dist.sample(sample_shape=n_particles)
#         new_particles = tf.gather(state.particles, indices, axis=1, batch_dims=1, validate_indices=False)
#
#         if self._stop_gradient:
#             new_particles = tf.stop_gradient(new_particles)
#
#         float_n_particles = tf.cast(n_particles, float)
#         uniform_weights = tf.ones_like(state.weights) / float_n_particles
#         uniform_log_weights = tf.zeros_like(state.log_weights) - tf.math.log(float_n_particles)
#
#         resampled_particles, resampled_weights, resampled_log_weights = resample(state.particles,
#                                                                                  new_particles,
#                                                                                  state.weights,
#                                                                                  uniform_weights,
#                                                                                  state.log_weights,
#                                                                                  uniform_log_weights,
#                                                                                  flags)
#
#         return attr.evolve(state, particles=resampled_particles, weights=resampled_weights,
#                            log_weights=resampled_log_weights)
