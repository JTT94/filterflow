import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

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
