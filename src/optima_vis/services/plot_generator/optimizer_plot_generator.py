from typing import Callable

import matplotlib.pyplot as plt
from torch import Tensor
from torch.nn import Module

from optima_vis.services.plot_generator.loss_contour_generator import (
    LossContourGenerator,
)


class OptimizerPlotGenerator:
    def __init__(self, loss_fcn: Callable[[Tensor, Tensor], Tensor]) -> None:
        self._loss_fcn = loss_fcn

    def __call__(
        self,
        model: Module,
        X_samples: Tensor,
        y_true: Tensor,
        loss_data: list[tuple[list[Tensor], Tensor]],
    ) -> None:
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

        plt.show()
