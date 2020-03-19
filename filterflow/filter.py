import tensorflow as tf

from filterflow.base import State, ObservationBase, InputsBase


class ParticleFilter(tf.Module):

    def __init__(self, observation_model, transition_model, emission_model, resampling_criterion, resampling_method,
                 name=None):
        super(ParticleFilter, self).__init__(name=name)
        self._observation_model = observation_model
        self._transition_model = transition_model
        self._emission_model = emission_model
        self._resampling_criterion = resampling_criterion
        self._resampling_method = resampling_method

    def propose(self, state: State, observation: ObservationBase, inputs: InputsBase):
        """Predict step of the filter

        :param state: State
            prior state of the filter
        :param observation: ObservationBase
            Observation used for look ahead proposal
        :param inputs: InputsBase
            Inputs used for prediction
        :return: Proposed State
        :rtype: State
        """

        # Resample the state
        # Use emitter
        # Apply transition loglikelihood

    def predict(self, state: State, inputs: InputsBase):
        """Predict step of the filter

        :param state: State
            prior state of the filter
        :param inputs: InputsBase
            Inputs used for preduction
        :return: Predicted State
        :rtype: State
        """
        # Use transition model

    def update_weights(self, proposed_state: State, observation: ObservationBase, inputs: InputsBase):
        """
        :param proposed_state:
            proposed state of the filter to be corrected
        :param observation: ObservationBase
            observation to compare the state against
        :param inputs: InputsBase
            inputs for the observation_model
        :return: Updated weights
        """
        # use emitted proposal
        # apply log likelihood of observation
