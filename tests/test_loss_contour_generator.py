from torch import Tensor, cat, rand
from torch.nn import Linear, MSELoss

from src.optima_vis.services.plot_generator.loss_contour_generator import (
    LossContourGenerator,
)


def test_mse_contour_generator() -> None:
    # Generate input features from interval [-10, 10].
    X_interval = [[-10, 10], [-10, 10]]
    X_1 = X_interval[0][0] + (X_interval[0][1] - X_interval[0][0]) * rand(100, 1)
    X_2 = X_interval[1][0] + (X_interval[1][1] - X_interval[1][0]) * rand(100, 1)
    X_samples = cat([X_1, X_2], 1)

    # Generate output features.
    def y_true_fcn(X: Tensor) -> Tensor:
        return (X @ Tensor([-5, 6.5]).unsqueeze(1)) + 4.0

    y_true = y_true_fcn(X_samples)

    # Make a linear regression model for testing.
    model = Linear(in_features=2, out_features=1, bias=False)
    loss_fcn = MSELoss()

    contour_generator = LossContourGenerator(loss_fcn, X_samples)

    contour_generator(model, y_true)
