from typing import Any, Callable

from numpy import inf
from torch import Tensor, cat, no_grad, rand, rand_like
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD

from src.internal.models.gif_properties import GifProperties
from src.internal.models.video_properties import VideoProperties
from src.internal.services.animation_generator.optimizer_animation_generator import (
    OptimizerAnimationGenerator,
)


def train(
    model: Module,
    X_samples: Tensor,
    y_true: Tensor,
    loss_fcn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Any,
    epochs: int = 1000,
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
        return X @ Tensor([-48.5, -32.5]).unsqueeze(1)

    y_true = y_true_fcn(X_samples)

    # Make a linear regression model for testing,
    # with custom weights for better plot experience
    model = Linear(in_features=2, out_features=1, bias=False)
    with no_grad():
        model.weight[:] = rand_like(model.weight)
    loss_fcn = MSELoss()
    optimizer = SGD(model.parameters(), lr=5e-3)

    # Train model
    loss_data = train(model, X_samples, y_true, loss_fcn, optimizer)

    optimizer_animation_generator = OptimizerAnimationGenerator()
    optimizer_animation_generator(
        model, X_samples, y_true, loss_fcn, loss_data, GifProperties()
    )
    optimizer_animation_generator(
        model, X_samples, y_true, loss_fcn, loss_data, VideoProperties()
    )
