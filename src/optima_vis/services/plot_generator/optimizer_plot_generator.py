from typing import Callable

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import Module

from optima_vis.services.plot_generator.loss_contour_generator import (
    LossContourGenerator,
)


class OptimizerPlotGenerator:
    """
    Generates static plots of optimization trajectories.

    This class overlays the path of parameter updates on top of a 2D loss
    contour plot. It is intended for models with exactly two trainable
    parameters, making it useful for visualizing how an optimizer navigates
    the loss surface.

    Example:
        >>> gen = OptimizerPlotGenerator(loss_fcn)
        >>> fig, ax = gen(model, X_samples, y_true, loss_data)
        >>> fig.show()
    """

    def __init__(self, loss_fcn: Callable[[Tensor, Tensor], Tensor]) -> None:
        """
        Initialize the plot generator.

        Args:
            loss_fcn (Callable[[Tensor, Tensor], Tensor]): Loss function used
                to evaluate the model’s predictions against targets.
        """

        self._loss_fcn = loss_fcn

    def __call__(
        self,
        model: Module,
        X_samples: Tensor,
        y_true: Tensor,
        loss_data: list[tuple[list[Tensor], Tensor]],
    ) -> tuple[Figure, Axes]:
        """
        Generate the optimization trajectory plot.

        Creates a contour plot of the loss surface and overlays the trajectory
        of parameter updates as red dots connected by arrows.

        Args:
            model (Module): PyTorch model with exactly two trainable parameters.
            X_samples (Tensor): Input samples used to compute the loss surface.
            y_true (Tensor): Ground-truth labels or targets.
            loss_data (list[tuple[list[Tensor], Tensor]]): Training history,
                where each element is a tuple ``(params, loss)`` containing:
                  - params (list[Tensor]): Model parameters at that iteration.
                  - loss (Tensor): Corresponding loss value.

        Returns:
            tuple[Figure, Axes]: Matplotlib figure and axes objects with the
            trajectory plot.
        """

        fig, ax = LossContourGenerator(self._loss_fcn, X_samples)(model, y_true)

        param_history: list[list[float]] = [
            sum((p.view(-1).tolist() for p in params), []) for params, _ in loss_data
        ]
        param_1: tuple[float]
        param_2: tuple[float]
        param_1, param_2 = zip(*param_history)

        ax.plot(param_1, param_2, "ro", markersize=4)

        for i in range(len(param_1) - 1):
            ax.annotate(
                "",
                xy=(param_1[i + 1], param_2[i + 1]),
                xytext=(param_1[i], param_2[i]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )

        return fig, ax
