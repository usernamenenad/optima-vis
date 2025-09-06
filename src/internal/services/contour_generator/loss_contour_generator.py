from typing import Any, Callable

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from torch import Tensor, cat, linspace, meshgrid, zeros_like, randn

from src.internal.models.neural_network_model import NeuralNetworkModel


class LossContourGenerator:
    def __init__(
        self,
        loss_fcn: Callable[[Tensor, Tensor], Tensor],
        X_intervals: list[Tensor | list[int]] = [[-10, 10], [-10, 10]],
        W_intervals: list[Tensor | list[int]] = [[-10, 10], [-10, 10]],
    ) -> None:
        N = 100

        X: list[Tensor] = []
        w: list[Tensor] = []

        for X_interval, W_interval in zip(X_intervals, W_intervals):
            X.append(X_interval[0] + (X_interval[1] - X_interval[0]) * randn(N, 1))
            w.append(linspace(W_interval[0], W_interval[1], N))

        self._X = cat(X, dim=1)
        self._w = meshgrid(w[0], w[1], indexing="ij")
        self._w_np = tuple([w.numpy() for w in self._w])

        self._loss_fcn = loss_fcn

    def __call__(
        self, model: NeuralNetworkModel, y_true: Callable[[Tensor], Tensor]
    ) -> Any:
        loss_mesh = zeros_like(self._w[0])

        print("Calculating losses...")
        for i in range(self._w[0].shape[0]):
            for j in range(self._w[0].shape[1]):
                y_pred = model.forward(
                    self._X, Tensor([self._w[0][i, j], self._w[1][i, j]])
                )
                loss_mesh[i, j] = self._loss_fcn(y_pred, y_true(self._X))
        print("Calculated all!")

        loss_mesh_np = loss_mesh.detach().numpy()

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(
            self._w_np[0], self._w_np[1], loss_mesh_np, levels=50, cmap="viridis"
        )
        plt.colorbar(contour)

        contour_levels = plt.contour(
            self._w_np[0],
            self._w_np[1],
            loss_mesh_np,
            levels=[10, 100, 500, 1000, 5000, 10000, 50000],
            colors="red",
            linewidths=1.5,
        )
        plt.clabel(contour_levels, inline=True, fontsize=8)

        plt.xlabel(r"Weight $w_1$")
        plt.ylabel(r"Weight $w_2$")

        plt.show()
