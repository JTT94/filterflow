import abc

from filterflow.base import Module


class TimeSeries(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def stack(self, observations):
        pass

    @abc.abstractmethod
    def next(self, inputs):
        pass