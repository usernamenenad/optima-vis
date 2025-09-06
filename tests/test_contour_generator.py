from torch import Tensor, matmul
from torch.nn import MSELoss

from src.internal.models.neural_network_model import NeuralNetworkModel
from src.internal.services.contour_generator.loss_contour_generator import (
    LossContourGenerator,
)


class LinearRegressionModel(NeuralNetworkModel):
    def forward(self, X: Tensor, w: Tensor) -> Tensor:
        return matmul(X, w) + 1.0


def test_mse_contour_generator() -> None:
    model = LinearRegressionModel()
    contour_generator = LossContourGenerator(MSELoss())

    def y_true(X: Tensor) -> Tensor:
        return matmul(X, Tensor([-2.5, 1.5])) + 1.0

    contour_generator(model, y_true)
