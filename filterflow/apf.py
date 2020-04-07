import attr
import tensorflow as tf

from filterflow.smc import SMC
from filterflow.base import State, Observation, InputsBase, Module, DTYPE_TO_STATE_SERIES, ObservationSeries
from filterflow.observation.base import ObservationModelBase
from filterflow.proposal.auxiliary_proposal import AuxiliaryProposal
from filterflow.resampling.base import ResamplerBase
from filterflow.resampling.criterion import ResamplingCriterionBase
from filterflow.transition.base import TransitionModelBase
from filterflow.utils import normalize


class AuxiliaryParticleFilter(SMC):
    def __init__(self, observation_model: ObservationModelBase, transition_model: TransitionModelBase,
                 proposal_model: AuxiliaryProposal, resampling_criterion: ResamplingCriterionBase,
                 resampling_method: ResamplerBase, name='SMC'):
        super(AuxiliaryParticleFilter, self).__init__(observation_model, transition_model,
                 proposal_model, resampling_criterion, resampling_method, name=name)



    