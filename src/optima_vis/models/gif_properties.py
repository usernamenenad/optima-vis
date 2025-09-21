from dataclasses import dataclass


@dataclass
class GifProperties:
    """
    Properties of GIF export.

    Properties
    ----------
    name : str
        The exported GIF's name.
    length : int
        The GIF length in seconds. Determined framerate of the GIF.
    """

    name: str = "_optimizer_animation"
    length: int = 10
