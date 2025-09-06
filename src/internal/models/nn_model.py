from abc import ABC, abstractmethod

from torch import Tensor


class NNModel(ABC):
    @abstractmethod
    def forward(
        self,
        X: Tensor,
        w: Tensor | None = None,
        b: Tensor | None = None,
    ) -> Tensor:
        pass
