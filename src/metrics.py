import tensorflow as tf
import keras

class PSNR(keras.metrics.Mean):
    def __init__(self, max_val: float = 1.0,
                 shave: int = 0, name: str = "psnr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.shave = shave

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                     sample_weight: tf.Tensor | None = None) -> tf.Tensor:
        if self.shave:
            s = self.shave
            y_true = y_true[:, s:-s, s:-s, :]
            y_pred = y_pred[:, s:-s, s:-s, :]
        psnr = tf.image.psnr(y_true, y_pred, self.max_val)
        super().update_state(psnr, sample_weight)
