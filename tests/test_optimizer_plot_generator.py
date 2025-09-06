from typing import Any, Callable, Iterator

from torch import Tensor, cat, rand
from torch.nn import Linear, MSELoss, Parameter
from torch.optim import SGD

from src.internal.models.nn_model import NNModel
from src.internal.services.plot_generator.optimizer_plot_generator import (
    OptimizerPlotGenerator,
)


class LinearRegressionModel(NNModel):
    def __init__(self, in_features: int, out_features: int) -> None:
        self._linear_model = Linear(in_features, out_features, bias=True)

    def forward(
        self, X: Tensor, w: Tensor | None = None, b: Tensor | None = None
    ) -> Tensor:
        if w is None or b is None:
            return self._linear_model(X)
        return X @ w.T + b

    def parameters(self) -> Iterator[Parameter]:
        return self._linear_model.parameters()

    def train(
        self,
        X_samples: Tensor,
        y_true: Tensor,
        loss_fcn: Callable[[Tensor, Tensor], Tensor],
        optimizer: Any,
        epochs: int = 1000,
        report_at_every: int = 10,
    ) -> list[tuple[Tensor, Tensor]]:
        loss_data: list[tuple[Tensor, Tensor]] = []

        for epoch in range(epochs):
            y_pred = self.forward(X_samples)

            loss = loss_fcn(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % report_at_every == 0:
                loss_data.append(
                    (
                        self._linear_model.weight.detach().clone(),
                        loss.detach().clone(),
                    )
                )

        return loss_data


def test_mse_contour_generator() -> None:
    # Generate input features from interval [-10, 10].
    X_interval = [[-10, 10], [-10, 10]]
    X_1 = X_interval[0][0] + (X_interval[0][1] - X_interval[0][0]) * rand(100, 1)
    X_2 = X_interval[1][0] + (X_interval[1][1] - X_interval[1][0]) * rand(100, 1)
    X_samples = cat([X_1, X_2], 1)

    # Generate output features.
    def y_true_fcn(X: Tensor) -> Tensor:
        return (X @ Tensor([-2.5, 1.5]).unsqueeze(1)) + 1.0

    y_true = y_true_fcn(X_samples)

    # Make a linear regression model for testing.
    model = LinearRegressionModel(in_features=2, out_features=1)
    loss_fcn = MSELoss()
    optimizer = SGD(model.parameters(), lr=1e-3)

    # Train it.
    loss_data = model.train(X_samples, y_true, loss_fcn, optimizer)

    # Plot loss movement.
    optimizer_plot_generator = OptimizerPlotGenerator(loss_fcn)
    optimizer_plot_generator(model, X_samples, y_true, loss_data)
