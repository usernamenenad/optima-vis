from typing import Callable

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor, linspace, meshgrid, no_grad, zeros_like
from torch.nn import Module


class LossContourGenerator:
    def __init__(
        self,
        loss_fcn: Callable[[Tensor, Tensor], Tensor],
        X: Tensor,
        param_intervals: list[Tensor] | list[list[float]] = [
            [-30.0, 30.0],
            [-30.0, 30.0],
        ],
    ) -> None:
        N = 100

        params: list[Tensor] = []

        for W_interval in param_intervals:
            params.append(linspace(W_interval[0], W_interval[1], N))

        self._X = X
        self._mesh_params = meshgrid(params[0], params[1], indexing="ij")
        self._mesh_params_np = tuple([param.numpy() for param in self._mesh_params])

        self._loss_fcn = loss_fcn

    def __call__(
        self,
        model: Module,
        y_true: Tensor,
    ) -> tuple[Figure, Axes]:
        loss_mesh = zeros_like(self._mesh_params[0])

        # Save original weights for restoring
        model_params = []
        for param in model.parameters():
            model_params.extend(param.view(-1))

        if len(model_params) != 2:
            raise ValueError(
                f"Model must have exactly 2 trainable degrees of freedom, got {len(model_params)}."
            )

        original_params = [param.clone() for param in model_params]

        with no_grad():
            for i in range(self._mesh_params[0].shape[0]):
                for j in range(self._mesh_params[0].shape[1]):
                    # Copy weights
                    model_params[0].copy_(self._mesh_params[0][i, j])
                    model_params[1].copy_(self._mesh_params[1][i, j])

                    # Calculate model output
                    y_pred = model(self._X)
                    loss_mesh[i, j] = self._loss_fcn(y_pred, y_true)

            # Restore original weights
            for param, original_param in zip(model_params, original_params):
                param.copy_(original_param)

        loss_mesh_np = loss_mesh.detach().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        contour = ax.contourf(
            self._mesh_params_np[0],
            self._mesh_params_np[1],
            loss_mesh_np,
            levels=25,
            cmap="Blues",
        )
        fig.colorbar(contour, ax=ax)

        contour_levels = ax.contour(
            self._mesh_params_np[0],
            self._mesh_params_np[1],
            loss_mesh_np,
            colors="dodgerblue",
            linewidths=1,
        )
        ax.clabel(contour_levels, inline=True, fontsize=8)

        ax.set_xlabel(r"Parameter $p_1$")
        ax.set_ylabel(r"Parameter $p_2$")

        return fig, ax
