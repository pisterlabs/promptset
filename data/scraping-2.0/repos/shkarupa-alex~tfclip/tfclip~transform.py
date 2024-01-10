import tensorflow as tf
from dataclasses import dataclass, asdict
from typing import Tuple, Union
from tfclip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from tfclip.utils import to_2tuple


@dataclass
class PreprocessCfg:
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'


_PREPROCESS_KEYS = set(asdict(PreprocessCfg()).keys())


def merge_preprocess_dict(base, overlay):
    if isinstance(base, PreprocessCfg):
        base_clean = asdict(base)
    else:
        base_clean = {k: v for k, v in base.items() if k in _PREPROCESS_KEYS}
    if overlay:
        overlay_clean = {k: v for k, v in overlay.items() if k in _PREPROCESS_KEYS and v is not None}
        base_clean.update(overlay_clean)
    return base_clean


def image_transform(image_size, interpolation_mode, resize_mode, name=None):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    def apply(image):
        with tf.name_scope(name or 'image_transform'):
            image = tf.convert_to_tensor(image)
            if image.shape.rank not in {3, 4} or 3 != image.shape[-1]:
                raise ValueError(f'Expecting a single or batched RGB image, got shape: {image.shape}')

            if 'bicubic' == interpolation_mode:
                interpolation = tf.image.ResizeMethod.BICUBIC
            elif 'bilinear' == interpolation_mode:
                interpolation = tf.image.ResizeMethod.BILINEAR
            else:
                raise ValueError(f'Unsupported interpolation mode: {interpolation_mode}')

            if 'squash' == resize_mode:
                image = tf.image.resize(image, image_size, method=interpolation, antialias=True)
            elif 'shortest' == resize_mode:
                source_size = tf.shape(image)
                source_size = source_size[:2] if 3 == image.shape.rank else source_size[1:3]
                source_size = tf.cast(source_size, 'float32')

                aspect_ratio = source_size / tf.cast(image_size, 'float32')
                aspect_ratio = tf.reduce_min(aspect_ratio)

                target_size = tf.cast(tf.round(source_size / aspect_ratio), 'int32')

                image = tf.image.resize(image, target_size, method=interpolation, antialias=True)
                image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])
            else:
                raise ValueError(f'Unsupported resize mode: {resize_mode}')

            image = tf.clip_by_value(image, 0, 255)
            image = tf.round(image)
            image = tf.cast(image, 'uint8')

            return image

    return apply
