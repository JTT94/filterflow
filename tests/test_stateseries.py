import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
import numpy as np

# import sys
# sys.path.append("./")

from filterflow.smc import SMC
from filterflow.base import State, ObservationBase, InputsBase, StateSeries
from filterflow.observation.linear import LinearObservation, LinearObservationModel, LinearObservationSeries
from filterflow.transition.random_walk import RandomWalkModel
from filterflow.proposal.base import BootstrapProposalModel
from filterflow.resampling.criterion import AlwaysResample
from filterflow.resampling.standard.multinomial import MultinomialResampler


class GHMMOutput(object):
    __slots__ = ('observations', 'states')
    def __init__(self, observations: np.ndarray, states: np.ndarray):
        self.observations = observations
        self.states = states

# Definition of the dynamic model
class GHMM(object):
    def __init__(self, 
                 initial_state: np.ndarray,
                 transition_matrix: np.ndarray, 
                 transition_covariance: np.ndarray,
                 observation_matrix: np.ndarray,
                 observation_covariance: np.ndarray,
                 seed: int = None):
        """
        Construction methods
        Parameters
        ----------

        """
        self.dim = initial_state.shape[0]
        self.initial_state = initial_state
                
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        
        self.observation_matrix = observation_matrix
        self.observation_covariance = observation_covariance
        
        self.random_seed = seed
                
    def sample_latent_state(self, T):
        latent_state = self.initial_state
        random_samples =  np.random.multivariate_normal(np.repeat(0., self.dim), self.transition_covariance, T)
        
        latent_state_ts = []
        latent_state_ts.append(latent_state)
        for i, random_sample in enumerate(random_samples):
            latent_state = self.transition_matrix @ latent_state
            latent_state += random_sample
            latent_state_ts.append(latent_state)
        return np.array(latent_state_ts)
        
    def get_observation(self, latent_space_ts):
        T = latent_space_ts.shape[0]
        no_noise = np.einsum('ij,kj->ki', self.observation_matrix, latent_space_ts)
        return no_noise + np.random.multivariate_normal(np.repeat(0., self.dim), self.observation_covariance, T)
    
    def sample(self, N: int) -> GHMMOutput:
        """
        Samples n steps of the model
        Parameters
        ----------
        N: int
            total number of steps
        
        Returns
        -------
        ResonatorOutput from the parametrised process
        """
        latent_time_series = self.sample_latent_state(N)
        observations = self.get_observation(latent_time_series)
        return GHMMOutput(observations, latent_time_series)

class TestStateSeries(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

        dimension = 2
        T = 150
        initial_state          = np.repeat(0., dimension)
        transition_matrix      = np.eye(dimension) * 0.5
        transition_covariance  = np.eye(dimension) * 0.5
        observation_matrix     = np.eye(dimension) *0.5
        observation_covariance = np.eye(dimension)*0.1

        linear_ss = GHMM(initial_state=initial_state, 
                        transition_matrix=transition_matrix,
                        transition_covariance=transition_covariance,
                        observation_matrix=observation_matrix,
                        observation_covariance=observation_covariance)
        ghmm_output = linear_ss.sample(T)

        batch_size = 1
        n_particles = 1000

        # observation
        observation_error = tfd.MultivariateNormalFullCovariance(np.array([0.,0.], dtype = np.float32),
                                                                linear_ss.observation_covariance.astype(np.float32))

        observation_model = LinearObservationModel(tf.constant(linear_ss.observation_matrix.astype(np.float32)), observation_error)

        # transition
        transition_noise = tfd.MultivariateNormalFullCovariance(np.array([0., 0.], dtype=np.float32), 
                                                            linear_ss.transition_covariance.astype(np.float32))

        transition_model = RandomWalkModel(tf.constant(linear_ss.transition_matrix.astype(np.float32)), 
                                        transition_noise)

        # proposal
        proposal_model = BootstrapProposalModel(transition_model)

        resampling_criterion = AlwaysResample()
        resampling_method = MultinomialResampler()

        particle_filter = SMC(observation_model, 
                            transition_model, 
                            proposal_model, 
                            resampling_criterion, 
                            resampling_method)

        # process observations
        observations = []
        for t, observation_value in enumerate(ghmm_output.observations):
            observations.append(LinearObservation(tf.constant(np.array([observation_value], dtype=np.float32))))

        observation_series = LinearObservationSeries(dtype = tf.float32, dimension = dimension)

        for t, observation in enumerate(observations):
            observation_series.write(t, observation)                      

        ## initial state
        weights = tf.ones((batch_size, n_particles), dtype= np.float32)/n_particles
        initial_particles = tf.random.uniform((batch_size, n_particles, dimension), -1, 1)
        log_likelihoods = tf.zeros((batch_size), dtype=float)
        state = State(batch_size,
                            n_particles,
                            dimension, 
                            initial_particles, 
                            log_weights= tf.math.log(weights),
                            weights=weights, 
                            log_likelihoods=log_likelihoods)

        # infer dimensions and type
        batch_size, n_particles, dimension = state.particles.shape
        dtype = state.particles.dtype
        self.state_series = StateSeries(dtype=dtype, batch_size=batch_size, n_particles=n_particles, dimension=dimension)
        self.states_list = []
        for t in range(observation_series.n_observations):
            obs = observation_series.read(t)
            new_state = particle_filter.update(state, obs, None)
            self.state_series.write(t, new_state)
            self.states_list.append(new_state)

            self.state_series.log_likelihoods_ta.mark_used()
            self.state_series.weights_ta.mark_used()
            self.state_series.log_weights_ta.mark_used()
            self.state_series.particles_ta.mark_used()

    def test_series(self):
        for t, state  in enumerate(self.states_list):
            recorded_state = self.state_series.read(t)
            self.assertAllEqual(state.particles, recorded_state.particles)
            self.assertAllEqual(state.log_likelihoods, recorded_state.log_likelihoods)
            self.assertAllEqual(state.log_weights, recorded_state.log_weights)

tf.test.main()