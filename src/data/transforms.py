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

def synthesize_lr(hr_image: tf.Tensor, 
                  scale: int, kernel: tf.Tensor) -> tf.Tensor:
    hr_image = gaussian_blur(hr_image, kernel)
    h, w = tf.shape(hr_image)[0], tf.shape(hr_image)[1]
    lr_hw = (h // scale, w // scale)
    lr_image = tf.image.resize(hr_image, lr_hw, method="bicubic")
    return tf.image.resize(lr_image, (h, w), "bicubic")

def random_patch_pair(hr_image: tf.Tensor, scale: int, fsub: int,
                      kernel: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    top = tf.random.uniform([], 0, tf.shape(hr_image)[0] - fsub + 1, tf.int32)
    left = tf.random.uniform([], 0, tf.shape(hr_image)[1] - fsub + 1, tf.int32)
    hr_patch = tf.image.crop_to_bounding_box(hr_image, top, left, fsub, fsub)
    lr_patch = synthesize_lr(hr_patch, scale, kernel)
    return lr_patch, hr_patch

def rgb_to_y(img: tf.Tensor) -> tf.Tensor:
    yuv = tf.image.rgb_to_yuv(img)
    return yuv[..., :1]

def crop_border(img: tf.Tensor, border: int) -> tf.Tensor:
    return img[border:-border, border:-border, :]

def augment_pair(lr: tf.Tensor,
                 hr: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    do_lr = tf.random.uniform([], 0, 1) < 0.5
    lr = tf.cond(do_lr, lambda: tf.image.flip_left_right(lr), lambda: lr)
    hr = tf.cond(do_lr, lambda: tf.image.flip_left_right(hr), lambda: hr)

    do_ud = tf.random.uniform([], 0, 1) < 0.5
    lr = tf.cond(do_ud, lambda: tf.image.flip_up_down(lr), lambda: lr)
    hr = tf.cond(do_ud, lambda: tf.image.flip_up_down(hr), lambda: hr)
    
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    lr = tf.image.rot90(lr, k)
    hr = tf.image.rot90(hr, k)
    return lr, hr


