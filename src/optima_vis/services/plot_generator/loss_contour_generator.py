from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor, linspace, meshgrid, no_grad, zeros_like
from torch.nn import Module


class LossContourGenerator:
    """
    Generates 2D contour plots of the loss surface for models with two trainable
    parameters.

    This class evaluates the loss function across a parameter grid and produces
    a contour plot. It is mainly useful for visualizing optimization dynamics
    in toy models that have exactly two degrees of freedom.

    Example:
        >>> gen = LossContourGenerator(loss_fcn, X)
        >>> fig, ax = gen(model, y_true)
        >>> fig.show()
    """

    def __init__(
        self,
        loss_fcn: Callable[[Tensor, Tensor], Tensor],
        X: Tensor,
        param_intervals: list[Tensor] | list[list[float]] = [
            [-30.0, 30.0],
            [-30.0, 30.0],
        ],
    ) -> None:
        """
        Initialize the loss contour generator.

        Args:
            loss_fcn (Callable[[Tensor, Tensor], Tensor]): Loss function that
                compares predictions and ground-truth labels.
            X (Tensor): Input samples used for evaluating the model.
            param_intervals (list[Tensor] | list[list[float]], optional):
                Ranges for the two model parameters. Defaults to
                ``[[-30.0, 30.0], [-30.0, 30.0]]``.
                Each inner list or tensor must specify [min, max] values
                for one parameter.

        Notes:
            - The generator creates a mesh grid of parameter values (100x100
              by default).
            - The model must have exactly two trainable parameters, otherwise
              an error is raised.
        """

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
        """
        Generate the loss contour plot.

        Evaluates the loss function over the 2D parameter grid and plots
        contour lines representing loss levels.

        Args:
            model (Module): PyTorch model with exactly two trainable parameters.
            y_true (Tensor): Ground-truth labels or targets.

        Returns:
            tuple[Figure, Axes]: Matplotlib figure and axes objects
            containing the contour plot.

        Raises:
            ValueError: If the model does not have exactly two trainable
            parameters.

        """

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

        contour_levels = ax.contour(
            self._mesh_params_np[0],
            self._mesh_params_np[1],
            loss_mesh_np,
            cmap="coolwarm",
            linewidths=1.5,
        )
        ax.clabel(contour_levels, inline=True, fontsize=8)

        ax.set_xlabel(r"Parameter $p_1$")
        ax.set_ylabel(r"Parameter $p_2$")

        return fig, ax
