from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from src.data.transforms import gaussian_kernel, synthesize_lr, random_patch_pair

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
        def _hr2lr(hr_path: Path) -> Path:
            return Path(self.config.dir, "LR",
                        hr_path.name.replace("_HR", "_LR"))
        lr_paths = [_hr2lr(p) for p in self.hr_paths]
        for lp in lr_paths:
            if not lp.exists():
                raise FileNotFoundError(f"No LP files: {lp}")

        hr_tensor = tf.convert_to_tensor([str(p) for p in self.hr_paths],
                                          tf.string)
        lr_tensor = tf.convert_to_tensor([str(p) for p in lr_paths], 
                                          tf.string)
        ds = tf.data.Dataset.from_tensor_slices((lr_tensor, hr_tensor))
        if self.config.use_lr:
            def _pair(lr_path, hr_path):
                lr = self._load(lr_path)
                hr = self._load(hr_path)
                hr_hw = tf.shape(hr)[:2]         
                lr_up = tf.image.resize(lr, hr_hw, "bicubic")
                return lr_up, hr
            ds = ds.map(_pair, tf.data.AUTOTUNE)
        else: 
            kernel, scale = self.kernel, self.config.scale
            ds = ds.map(lambda _lr, hr: (synthesize_lr(self._load(hr),
                                                  scale, kernel),
                                    self._load(hr)), tf.data.AUTOTUNE)
        return ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        