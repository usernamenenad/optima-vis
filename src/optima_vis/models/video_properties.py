from dataclasses import dataclass


@dataclass
class VideoProperties:
    """
    Properties of video type of export.

    Properties
    ----------
    name : str
        The exported video's name.
    length : int
        The video length in seconds. Determined framerate of the video. Defaults to `10s`.
    format: str
        The exported video's format. Defaults to `mp4`.
    """

    name: str = "_optimizer_animation"
    length: int = 10
    format: str = "mp4"
