from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from transforms import gaussian_kernel, synthesize_lr, random_patch_pair

@dataclass(slots=True)
class TrainsetConfig:
    dir: Path
    scale: int = 2
    fsub: int = 33
    batch_size: int = 16
    shuffle_buffer: int = 400


@dataclass
class TrainDataset:
    config: TrainsetConfig
    
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

    def build(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(self.paths).repeat()
        ds = ds.map(self._load, tf.data.AUTOTUNE)
        config, kernel = self.config, self.kernel
        ds = ds.map(lambda hr_image: random_patch_pair(hr_image, config.scale,
                                                       config.fsub, kernel),
                                                        tf.data.AUTOTUNE)
        ds = ds.shuffle(config.shuffle_buffer).batch(config.batch_size,
                                                     drop_remainder=True)
        return ds.prefetch(tf.data.AUTOTUNE)
    

@dataclass(slots=True)
class PairedConfig:
    dir: Path
    use_lr: bool = True
    scale: int = 2
    batch_size: int = 1


@dataclass
class PairedDataset:
    config: PairedConfig

    def __post_init__(self):
        self.kernel = gaussian_kernel(kernel_size=13, sigma=1.6)
        hr_dir = Path(self.config.dir, "HR")
        lr_dir = Path(self.config.dir, "LR")
        self.hr_paths = sorted(hr_dir.glob("*"))
        assert self.hr_paths, "Files HR not found in {hr_dir}"
        if self.config.use_lr and not list(lr_dir.glob("*")):
            raise FileNotFoundError("Files LR not found in {lr_dir}")
    
    @staticmethod
    def _load(path: tf.Tensor) -> tf.Tensor:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, 3, expand_animations=False)
        return tf.image.convert_image_dtype(img, tf.float32)
    
    def build(self) -> tf.data.Dataset:
        lr_paths = [Path(self.config.dir, "LR", p.name) for p in self.hr_paths]
        hr_tensor = tf.convert_to_tensor([str(p) for p in self.hr_paths],
                                          tf.string)
        lr_tensor = tf.convert_to_tensor([str(p) for p in lr_paths],
                                          tf.string)
        ds = tf.data.Dataset.from_tensor_slices((lr_tensor, hr_tensor))
        if self.config.use_lr:
            ds = ds.map(lambda lr, hr: (self._load(lr), self._load(hr)),
                        tf.data.AUTOTUNE)
        else:
            kernel, scale = self.kernel, self.config.scale
            ds = ds.map(lambda hr: (synthesize_lr(self._load(hr),
                                                  scale, kernel),
                                    self._load(hr), tf.data.AUTOTUNE))
        return ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        