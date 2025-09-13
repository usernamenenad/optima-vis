from dataclasses import dataclass


@dataclass
class VideoProperties:
    name: str = "_optimizer_animation"
    length: int = 10
    format: str = "mp4"
