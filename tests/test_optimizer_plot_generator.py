from typing import Any, Callable

from numpy import inf
from torch import Tensor, cat, rand
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD

from src.optima_vis.services.plot_generator.optimizer_plot_generator import (
    OptimizerPlotGenerator,
)


def train(
    model: Module,
    X_samples: Tensor,
    y_true: Tensor,
    loss_fcn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Any,
    epochs: int = 2000,
    report_at_every: int = 10,
) -> list[tuple[list[Tensor], Tensor]]:
    loss_data: list[tuple[list[Tensor], Tensor]] = [
        (
            [w[0].clone() for w in list(model.parameters())],
            Tensor([inf]),
        )
    ]

    for epoch in range(epochs):
        y_pred = model(X_samples)

        loss = loss_fcn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % report_at_every == 0:
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

    # Generate output features
    def y_true_fcn(X: Tensor) -> Tensor:
        return X @ Tensor([6.5, 4.5]).unsqueeze(1)

    y_true = y_true_fcn(X_samples)

    # Make a linear regression model for testing
    model = Linear(in_features=2, out_features=1, bias=False)
    loss_fcn = MSELoss()
    optimizer = SGD(model.parameters(), lr=5e-3)

    # Train model
    loss_data = train(model, X_samples, y_true, loss_fcn, optimizer)

    # Plot loss movement
    optimizer_plot_generator = OptimizerPlotGenerator(loss_fcn)
    optimizer_plot_generator(model, X_samples, y_true, loss_data)


def test_mse_contour_generator_with_bias() -> None:
    # Generate input features from interval [-10, 10]
    X_interval = [[-10, 10], [-10, 10]]
    X_samples = X_interval[0][0] + (X_interval[0][1] - X_interval[0][0]) * rand(100, 1)

    # Generate output features
    def y_true_fcn(X: Tensor) -> Tensor:
        return X @ Tensor([-6.5]).unsqueeze(1) + 4.0

    y_true = y_true_fcn(X_samples)

    # Make a linear regression model for testing
    model = Linear(in_features=1, out_features=1, bias=True)
    loss_fcn = MSELoss()
    optimizer = SGD(model.parameters(), lr=5e-3)

    # Train model
    loss_data = train(model, X_samples, y_true, loss_fcn, optimizer)

    # Plot loss movement
    optimizer_plot_generator = OptimizerPlotGenerator(loss_fcn)
    optimizer_plot_generator(model, X_samples, y_true, loss_data)
