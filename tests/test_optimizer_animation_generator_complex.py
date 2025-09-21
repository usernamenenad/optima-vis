from typing import Any, Callable

from numpy import inf
from torch import Tensor, cat, rand, randn_like, tensor
from torch.nn import Module, MSELoss, Parameter
from torch.optim import RMSprop, SGD

from optima_vis.models.gif_properties import GifProperties
from optima_vis.services.animation_generator.optimizer_animation_generator import (
    OptimizerAnimationGenerator,
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
    epochs: int = 200,
    report_at_every: int = 1,
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
            print(f"[{epoch + 1}] Current loss: {loss.data}.")

            loss_data.append(
                (
                    [w[0].clone() for w in list(model.parameters())],
                    loss.detach().clone(),
                )
            )

        if loss.item() < 1e-5:
            break

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
    optimizer = SGD(model.parameters(), lr=5e-2)
    # optimizer = RMSprop(model.parameters(), lr=2, alpha=0.999)

    # Train model
    loss_data = train(model, X_samples, y_measured, loss_fcn, optimizer)

    optimizer_animation_generator = OptimizerAnimationGenerator()
    optimizer_animation_generator(
        model, X_samples, y_true, loss_fcn, loss_data, GifProperties()
    )
    # optimizer_animation_generator(
    #     model, X_samples, y_measured, loss_fcn, loss_data, VideoProperties()
    # )
