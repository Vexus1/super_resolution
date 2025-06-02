from typing import Optional

import tensorflow as tf
import keras

class PSNR(keras.metrics.Mean):
    def __init__(self, max_val: float = 1.0, name: str = "psnr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = max_val

    def update_state(self, y_true: tf.Tensor,
                     y_pred: tf.Tensor,
                     sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
        psnr = tf.image.psnr(y_true, y_pred, max_val=self.max_val)
        psnr = tf.image.psnr(y_true, y_pred, self.max_val)
        return super().update_state(psnr, sample_weight)
