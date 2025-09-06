from abc import ABC, abstractmethod

from torch import Tensor


class NeuralNetworkModel(ABC):
    @abstractmethod
    def forward(self, X: Tensor, w: Tensor) -> Tensor:
        pass
