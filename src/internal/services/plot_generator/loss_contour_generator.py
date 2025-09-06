from typing import Callable

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor, linspace, meshgrid, zeros_like

from src.internal.models.nn_model import NNModel


class LossContourGenerator:
    def __init__(
        self,
        loss_fcn: Callable[[Tensor, Tensor], Tensor],
        X_samples: Tensor,
        W_intervals: list[Tensor | list[float]] = [[-5.0, 5.0], [-5.0, 5.0]],
    ) -> None:
        N = 100

        w: list[Tensor] = []

        for W_interval in W_intervals:
            w.append(linspace(W_interval[0], W_interval[1], N))

        self._X = X_samples
        self._w = meshgrid(w[0], w[1], indexing="ij")
        self._w_np = tuple([w.numpy() for w in self._w])

        self._loss_fcn = loss_fcn

    def __call__(
        self,
        model: NNModel,
        y_true: Tensor,
    ) -> tuple[Figure, Axes]:
        loss_mesh = zeros_like(self._w[0])

        print("Calculating losses...")
        for i in range(self._w[0].shape[0]):
            for j in range(self._w[0].shape[1]):
                y_pred = model.forward(
                    self._X,
                    Tensor([self._w[0][i, j], self._w[1][i, j]]),
                    Tensor([1.0]),
                )
                loss_mesh[i, j] = self._loss_fcn(y_pred, y_true)
        print("Calculated all!")

        loss_mesh_np = loss_mesh.detach().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        contour = ax.contourf(
            self._w_np[0], self._w_np[1], loss_mesh_np, levels=10, cmap="Blues"
        )
        fig.colorbar(contour, ax=ax)

        contour_levels = ax.contour(
            self._w_np[0],
            self._w_np[1],
            loss_mesh_np,
            levels=[10, 100, 500, 1000, 5000, 10000, 50000],
            colors="red",
            linewidths=1.5,
        )
        ax.clabel(contour_levels, inline=True, fontsize=8)

        ax.set_xlabel(r"Weight $w_1$")
        ax.set_ylabel(r"Weight $w_2$")

        return fig, ax
