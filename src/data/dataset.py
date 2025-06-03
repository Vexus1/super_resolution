from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from src.data.transforms import *

BORDER = 6 # 9-1-5

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
        def _make_pair(hr_img):
            lr_p, hr_p = random_patch_pair(hr_img, config.scale,
                                           config.fsub, kernel)
            lr_p, hr_p = augment_pair(lr_p, hr_p)
            lr_p = rgb_to_y(lr_p)                          # [33, 33, 1]
            hr_p = crop_border(rgb_to_y(hr_p), BORDER)     # [21, 21, 1]
            return lr_p, hr_p

        ds = ds.map(_make_pair, tf.data.AUTOTUNE)
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
            def _pair(lr_p, hr_p):
                lr = rgb_to_y(self._load(lr_p))
                hr = rgb_to_y(self._load(hr_p))
                hr_hw = tf.shape(hr)[:2]
                lr = tf.image.resize(lr, hr_hw, "bicubic") 
                return lr, crop_border(hr, BORDER)
            ds = ds.map(_pair, tf.data.AUTOTUNE)
        else:
            kernel, scale = self.kernel, self.config.scale
            ds = ds.map(
                lambda _lr, hr_p: (
                    synthesize_lr(rgb_to_y(self._load(hr_p)), scale, kernel),
                    crop_border(rgb_to_y(self._load(hr_p)), BORDER)),
                    tf.data.AUTOTUNE
                )
        return ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        