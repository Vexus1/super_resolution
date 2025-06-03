from dataclasses import dataclass
from pathlib import Path

@dataclass(slots=True, frozen=True)
class ModelHP:
    filters: tuple[int, int, int]       = (128, 64, 1)
    kernel_sizes: tuple[int, int, int]  = (9, 1, 5)
    input_channels: int                 = 1


@dataclass(slots=True, frozen=True)
class TrainHP:
    epochs: int             = 200
    steps_per_epoch: int    = 1000
    batch_size: int         = 128
    lr_init: float          = 1e-4
    lr_factor: float        = 0.5
    lr_patience: int        = 5
    lr_min: float           = 1e-6


@dataclass(slots=True, frozen=True)
class DataHP:
    root: Path              = Path("dataset")
    scale: int              = 2
    fsub: int               = 33
    shuffle_buffer: int     = 400


@dataclass(slots=True, frozen=True)
class BlurHP:
    kernel_size: int        = 13
    sigma: float            = 1.6
    border: int             = 6
