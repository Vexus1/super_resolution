from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from transforms import gaussian_kernel, synthesize_lr, random_patch_pair

@dataclass(slots=True)
class TrainSetConfig:
    dir: Path
    scale: int = 2
    fsub: int = 33
    batch_size: int = 16
    shuffle_buffer: int = 400


@dataclass
class TrainDataSet:
    config: TrainSetConfig
    
    def __post_init__(self):
        self.kernel = gaussian_kernel(kernel_size=13, sigma=1.6)
        self.paths = self._get_paths()
    
    def _get_paths(self) -> tf.Tensor:
        paths = sorted(Path(self.config.dir).glob("*.png"))
        assert len(paths) == 91, "Required 91 files from the T91 test set"
        return tf.convert_to_tensor([str(p) for p in paths], tf.string)
    
    @staticmethod
    def _load(path: tf.Tensor) -> tf.Tensor:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        return tf.image.convert_image_dtype(img, tf.float32)
