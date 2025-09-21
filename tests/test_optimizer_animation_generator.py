from typing import Any, Callable

from numpy import inf
from torch import Tensor, cat, no_grad, rand, rand_like, randn_like
from torch.nn import Linear, Module, MSELoss
from torch.optim import RMSprop, SGD

from optima_vis.models.video_properties import VideoProperties
from optima_vis.models.gif_properties import GifProperties
from optima_vis.services.animation_generator.optimizer_animation_generator import (
    OptimizerAnimationGenerator,
)


def train(
    model: Module,
    X_samples: Tensor,
    y_true: Tensor,
    loss_fcn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Any,
    epochs: int = 100,
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
        return X @ Tensor([48.5, -0.5]).unsqueeze(1)

    y_true = y_true_fcn(X_samples)
    y_measured = y_true + 2.0 * randn_like(y_true)

    # Make a linear regression model for testing,
    # with custom weights for better plot experience
    model = Linear(in_features=2, out_features=1, bias=False)
    with no_grad():
        model.weight[:] = Tensor([[-15.5, -15.5]])
    loss_fcn = MSELoss()
    # optimizer = SGD(model.parameters(), lr=5e-2)
    optimizer = RMSprop(model.parameters(), lr=5e-1)

    # Train model
    loss_data = train(model, X_samples, y_measured, loss_fcn, optimizer)

    optimizer_animation_generator = OptimizerAnimationGenerator()
    optimizer_animation_generator(
        model, X_samples, y_true, loss_fcn, loss_data, GifProperties()
    )
    # optimizer_animation_generator(
    #     model, X_samples, y_measured, loss_fcn, loss_data, VideoProperties()
    # )
