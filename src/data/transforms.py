import tensorflow as tf

def gaussian_kernel(kernel_size: int = 13, sigma: float = 1.6) -> tf.Tensor:
    ax = tf.cast(tf.range(kernel_size), tf.float32) - (kernel_size - 1) / 2.0
    g = tf.exp(-(ax**2) / (2.0 * sigma**2))
    k = tf.tensordot(g, g, axes=0)
    k /= tf.reduce_sum(k)
    return k[:, :, None, None]  # [k,k,1,1]

def gaussian_blur(img: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    img = img[None, ...]
    kernel = tf.tile(kernel, [1, 1, tf.shape(img)[-1], 1])
    blur_img = tf.nn.depthwise_conv2d(img, kernel, [1, 1, 1, 1], "SAME")
    # blur_img = tf.cast(blur_img, tf.uint8)
    return blur_img[0]
