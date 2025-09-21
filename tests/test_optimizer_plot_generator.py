from typing import Any, Callable

import matplotlib.pyplot as plt
from numpy import inf
from torch import Tensor, cat, rand, randn_like, tensor
from torch.nn import Module, MSELoss, Parameter
from torch.optim import Adam

from src.optima_vis.services.plot_generator.optimizer_plot_generator import (
    OptimizerPlotGenerator,
)


class Complex2DModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = Parameter(tensor([-3.0]))
        self.w2 = Parameter(tensor([1.0]))

    def forward(self, X: Tensor):
        return self.w1 * X[:, 0:1] ** 2 + self.w2 * X[:, 1:2] ** 2


def train(
    model: Module,
    X_samples: Tensor,
    y_true: Tensor,
    loss_fcn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Any,
    epochs: int = 300,
    report_at_every: int = 1,
) -> list[tuple[list[Tensor], Tensor]]:
    prev_loss = Tensor([inf])
    tol = 1e-5

    loss_data: list[tuple[list[Tensor], Tensor]] = [
        (
            [w[0].clone() for w in list(model.parameters())],
            prev_loss,
        )
    ]

    for epoch in range(epochs):
        y_pred = model(X_samples)

        loss = loss_fcn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if abs(prev_loss - loss.item()) < tol:
            print(f"[{epoch + 1}] Stopping early, loss change < {tol}")
            break

        prev_loss = loss.item()

        if (epoch + 1) % report_at_every == 0:
            print(f"[{epoch + 1}] Loss: {loss.data}")
            loss_data.append(
                (
                    [w[0].clone() for w in list(model.parameters())],
                    loss.detach().clone(),
                )
            )

    return loss_data


def test_mse_contour_generator_without_bias() -> None:
    # Generate input features from interval [-10, 10]
    X_interval = [[-10, 10], [-10, 10]]
    X_1 = X_interval[0][0] + (X_interval[0][1] - X_interval[0][0]) * rand(100, 1)
    X_2 = X_interval[1][0] + (X_interval[1][1] - X_interval[1][0]) * rand(100, 1)
    X_samples = cat([X_1, X_2], 1)
    X_samples = (X_samples - X_samples.mean(0)) / X_samples.std(0)

    # Generate output features
    def y_true_fcn(X: Tensor) -> Tensor:
        return -43.4 * X[:, 0:1] ** 2 + -32.5 * X[:, 1:2] ** 2

    y_true = y_true_fcn(X_samples)
    y_measured = y_true + 4 * randn_like(y_true)

    # Make a linear regression model for testing,
    # with custom weights for better plot experience
    model = Complex2DModel()
    loss_fcn = MSELoss()
    # optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.7, nesterov=True)
    optimizer = Adam(model.parameters(), lr=1e-1)

    # Train model
    loss_data = train(model, X_samples, y_measured, loss_fcn, optimizer)

    # Plot loss movement
    optimizer_plot_generator = OptimizerPlotGenerator(loss_fcn)
    fig, ax = optimizer_plot_generator(model, X_samples, y_true, loss_data)

    plt.show()


# def test_mse_contour_generator_with_bias() -> None:
#     # Generate input features from interval [-10, 10]
#     X_interval = [[-10, 10], [-10, 10]]
#     X_samples = X_interval[0][0] + (X_interval[0][1] - X_interval[0][0]) * rand(100, 1)

#     # Generate output features
#     def y_true_fcn(X: Tensor) -> Tensor:
#         return X @ Tensor([-6.5]).unsqueeze(1) + 4.0

#     y_true = y_true_fcn(X_samples)

#     # Make a linear regression model for testing
#     model = Linear(in_features=1, out_features=1, bias=True)
#     loss_fcn = MSELoss()
#     optimizer = SGD(model.parameters(), lr=5e-3)

#     # Train model
#     loss_data = train(model, X_samples, y_true, loss_fcn, optimizer)

#     # Plot loss movement
#     optimizer_plot_generator = OptimizerPlotGenerator(loss_fcn)
#     optimizer_plot_generator(model, X_samples, y_true, loss_data)
