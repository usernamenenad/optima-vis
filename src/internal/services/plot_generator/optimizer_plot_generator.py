from typing import Callable

import matplotlib.pyplot as plt
from torch import Tensor

from src.internal.models.nn_model import NNModel
from src.internal.services.plot_generator.loss_contour_generator import (
    LossContourGenerator,
)


class OptimizerPlotGenerator:
    def __init__(self, loss_fcn: Callable[[Tensor, Tensor], Tensor]) -> None:
        self._loss_fcn = loss_fcn

    def __call__(
        self,
        model: NNModel,
        X_samples: Tensor,
        y_true: Tensor,
        loss_data: list[tuple[Tensor, Tensor]],
    ) -> None:
        fig, ax = LossContourGenerator(self._loss_fcn, X_samples)(model, y_true)
        w1 = [w[0][0].item() for w, _ in loss_data]
        w2 = [w[0][1].item() for w, _ in loss_data]

        ax.plot(w1, w2, "ro", markersize=4)

        for i in range(len(w1) - 1):
            ax.annotate(
                "",
                xy=(w1[i + 1], w2[i + 1]),
                xytext=(w1[i], w2[i]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )

        plt.show()
