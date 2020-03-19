import tensorflow as tf

from filterflow.base import State, ObservationBase

class ParticleFilter(tf.Module):

    def __init__(self, observation_model, transition_model, emission_model, resampling_criterion, resampling_method,
                 name=None):
        super(ParticleFilter, self).__init__(name=name)
        self._observation_model = observation_model
        self._transition_model = transition_model
        self._emission_model = emission_model
        self._resampling_criterion = resampling_criterion
        self._resampling_method = resampling_method



