from typing import Callable

from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from torch import Tensor
from torch.nn import Module

from optima_vis.models.gif_properties import GifProperties
from optima_vis.models.video_properties import VideoProperties
from optima_vis.services.plot_generator.loss_contour_generator import (
    LossContourGenerator,
)


class OptimizerAnimationGenerator:
    """
    Generates optimization trajectory animations (GIF or MP4).

    This class creates animations of parameter updates during optimization
    by plotting their trajectory over a loss contour surface. It can export
    animations as either video (MP4) or GIF depending on the provided
    export properties.

    Example:
        >>> gen = OptimizerAnimationGenerator()
        >>> gen(
        ...     model,
        ...     X_samples,
        ...     y_true,
        ...     loss_fcn,
        ...     loss_data,
        ...     VideoProperties(name="training", length=5, format="mp4"),
        ... )
        GENERATING VIDEO...
    """

    @staticmethod
    def __update(
        param_1: tuple[float],
        param_2: tuple[float],
        line: Line2D,
        frame: int,
        ax: Axes,
        arrow: FancyArrowPatch | None = None,
    ) -> tuple[Line2D]:
        """
        Update function for each animation frame.

        Plots the optimization trajectory by updating the line and arrow
        showing parameter movement across frames.

        Args:
            param_1 (tuple[float]): History of first parameter values.
            param_2 (tuple[float]): History of second parameter values.
            line (Line2D): Line object representing the optimization path.
            frame (int): Current frame index.
            ax (Axes): Matplotlib Axes object to draw on.
            arrow (FancyArrowPatch | None, optional): Arrow showing the last
                parameter update. Defaults to None.

        Returns:
            tuple[Line2D]: Updated line object as required by
            :class:`matplotlib.animation.FuncAnimation`.
        """

        p1 = param_1[: frame + 1]
        p2 = param_2[: frame + 1]

        line.set_data([p1], [p2])

        if arrow is not None:
            arrow.remove()

        if frame > 0:
            arrow = FancyArrowPatch(
                (p1[-2], p2[-2]),
                (p1[-1], p2[-1]),
                arrowstyle="->",
                color="blue",
                mutation_scale=12,
                lw=1.5,
            )
            ax.add_patch(arrow)

        return (line,)

    def __call__(
        self,
        model: Module,
        X_samples: Tensor,
        y_true: Tensor,
        loss_fcn: Callable[[Tensor, Tensor], Tensor],
        loss_data: list[tuple[list[Tensor], Tensor]],
        export_properties: VideoProperties | GifProperties,
    ) -> None:
        """
        Generate and export the optimization animation.

        Builds the loss contour plot, overlays the optimization trajectory,
        and creates an animated visualization. The animation is exported
        as either MP4 (video) or GIF depending on the provided
        export properties.

        Args:
            model (Module): The PyTorch model being optimized.
            X_samples (Tensor): Input samples used to compute loss contours.
            y_true (Tensor): Ground-truth labels or targets.
            loss_fcn (Callable[[Tensor, Tensor], Tensor]): Loss function
                comparing predictions and targets.
            loss_data (list[tuple[list[Tensor], Tensor]]): History of
                parameter states and their associated loss values across
                training iterations.
            export_properties (VideoProperties | GifProperties): Export
                configuration specifying output name, format, and duration.

        Raises:
            Exception: If the export format is not recognized.

        """

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
            loss_fcn,
            X_samples,
            param_intervals,
        )(model, y_true)

        (line,) = ax.plot([], [], "bo-", markersize=3)

        def __init() -> tuple[Line2D]:
            line.set_data([], [])
            return (line,)

        arrow: FancyArrowPatch | None = None
        ani = FuncAnimation(
            fig=fig,
            init_func=__init,
            func=lambda frame: OptimizerAnimationGenerator.__update(
                param_1,
                param_2,
                line,
                frame,
                ax=ax,
                arrow=arrow,
            ),
            frames=len(param_1),
            blit=False,
            interval=1000,
            repeat=False,
        )

        if isinstance(export_properties, VideoProperties):
            print("\033[32mGENERATING VIDEO...\033[0m")
            # Determine fps
            fps = len(loss_data) // export_properties.length

            # Save animation as .mp4
            ani.save(
                f"{export_properties.name}.{export_properties.format}",
                writer="ffmpeg",
                fps=fps,
            )
        elif isinstance(export_properties, GifProperties):
            print("\033[32mGENERATING GIF...\033[0m")
            # Determine fps
            fps = len(loss_data) // export_properties.length

            # Save animation as .gif
            ani.save(
                f"{export_properties.name}.gif",
                writer="pillow",
                fps=fps,
            )
        else:
            raise Exception("Unknown export format!")
