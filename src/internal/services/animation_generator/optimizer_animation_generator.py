from typing import Callable

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from torch import Tensor
from torch.nn import Module

from src.internal.services.plot_generator.loss_contour_generator import (
    LossContourGenerator,
)


class OptimizerAnimationGenerator:
    def __init__(self, loss_fcn: Callable[[Tensor, Tensor], Tensor]) -> None:
        self._loss_fcn = loss_fcn

    @staticmethod
    def __update(
        param_1: tuple[float],
        param_2: tuple[float],
        line: Line2D,
        frame: int,
    ) -> tuple[Line2D]:
        line.set_data([param_1[: frame + 1]], [param_2[: frame + 1]])
        return (line,)

    def __call__(
        self,
        model: Module,
        X_samples: Tensor,
        y_true: Tensor,
        loss_data: list[tuple[list[Tensor], Tensor]],
        video_length: int | None = 10,
    ) -> None:
        param_history: list[list[float]] = [
            sum((p.view(-1).tolist() for p in params), []) for params, _ in loss_data
        ]

        param_1: tuple[float]
        param_2: tuple[float]
        param_1, param_2 = zip(*param_history)

        param_intervals = [
            [max(param_1) + 10, min(param_1) - 10],
            [max(param_2) + 10, min(param_2) - 10],
        ]

        fig, ax = LossContourGenerator(
            self._loss_fcn,
            X_samples,
            param_intervals,
        )(model, y_true)

        (line,) = ax.plot([], [], "ro-", markersize=5)

        def __init() -> tuple[Line2D]:
            line.set_data([], [])
            return (line,)

        ani = FuncAnimation(
            fig=fig,
            init_func=__init,
            func=lambda frame: OptimizerAnimationGenerator.__update(
                param_1, param_2, line, frame
            ),
            frames=len(param_1),
            blit=True,
            interval=1000,
            repeat=False,
        )

        # Determine fps
        if not video_length:
            video_length = 10
        fps = len(loss_data) // video_length

        # Save video as .mp4
        ani.save("neso.mp4", writer="ffmpeg", fps=fps)
