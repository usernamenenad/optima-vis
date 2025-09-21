# Optima Vis

**Optima Vis** is a Python package for visualizing optimization trajectories of PyTorch models in 2D parameter space.  
It provides tools to plot loss surfaces and generate static or animated visualizations of optimizer behavior, either as GIFs or MP4 videos.

## Features

- **LossContourGenerator** – Generates 2D contour plots of the loss surface for models with exactly two trainable parameters.
- **OptimizerPlotGenerator** – Creates static plots of optimization trajectories on top of the loss contours.
- **OptimizerAnimationGenerator** – Generates animated visualizations (GIF/MP4) of the optimizer's path through parameter space.

## Installation

You can install via PyPI (once published) or directly from source:

```bash
# From PyPI
pip install optima-vis

# From source
git clone https://github.com/yourusername/optima-vis.git
cd optima-vis
pip install .
```

## Dependencies:

- Python 3.13+
- PyTorch 2.8+
- Matplotlib 3.10+

## Usage

```python
from torch import nn, tensor
from torch import randn_like
from optima_vis import (
    OptimizerPlotGenerator,
    OptimizerAnimationGenerator,
    LossContourGenerator,
    VideoProperties,
    GifProperties
)

```

### Generate a loss contour

```python
loss_fcn = nn.MSELoss()
X_interval = [[-10, 10], [-10, 10]]
X_1 = X_interval[0][0] + (X_interval[0][1] - X_interval[0][0]) * rand(100, 1)
X_2 = X_interval[1][0] + (X_interval[1][1] - X_interval[1][0]) * rand(100, 1)
X_samples = cat([X_1, X_2], 1)

def y_true_fcn(X: Tensor) -> Tensor:
    return (X @ Tensor([-5, 6.5]).unsqueeze(1))

y_true = y_true_fcn(X_samples)
y_measured = y_true + 2.0 * randn_like(y_true)  # Measurements with noise

model = Linear(in_features=2, out_features=1, bias=False)

# Create contour generator
contour_gen = LossContourGenerator(loss_fcn, X_samples)
fig, ax = contour_gen(model, y_true)
fig.show()
```

### Plot optimization trajectory

```python
optimizer_plot_gen = OptimizerPlotGenerator(loss_fcn)
fig, ax = optimizer_plot_gen(model, X_samples, y_true, loss_data)
fig.show()
```

- `loss_data` is a list of tuples: (`list[Tensor], Tensor`) representing parameters and corresponding loss at each iteration.

### Generate animated visualization

```python
animation_gen = OptimizerAnimationGenerator()

# Export as GIF
export_props = GifProperties(name="training_animation", length=5)
animation_gen(model, X_samples, y_true, loss_fcn, loss_data, export_props)

# Export as MP4 video
export_props = VideoProperties(name="training_video", length=5, format="mp4")
animation_gen(model, X_samples, y_true, loss_fcn, loss_data, export_props)

```

- Displays the optimizer's path on the loss surface.
- Supports both GIF and MP4 export formats.
- Uses matplotlib.animation.FuncAnimation internally.

# Notes

- Only supports models with exactly two trainable parameters.
- For animations, the frame rate is automatically calculated based on the total number of iterations and the desired video length.
- Arrows indicate the direction of parameter updates in the trajectory.
- More examples can be found under `tests` folder.
