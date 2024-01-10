#!/usr/bin/env python
"""Pretrain the visual component."""
import os
import sys

import keras_cortex

from angorapy.models.convolutional import OpenAIEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math

from angorapy.utilities.util import mpi_flat_print

import numpy as np
import tensorflow_datasets as tfds
from tensorflow.python.data import AUTOTUNE

from angorapy.models import _build_openai_encoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Tuple

import argcomplete
import tensorflow as tf

from angorapy.common.const import PRETRAINED_COMPONENTS_PATH, VISION_WH
from angorapy.utilities.data_generation import gen_cube_quats_prediction_data, load_dataset

import tensorflow_graphics.geometry.transformation as tfg


def rotational_diff_metric(y_true, y_pred):
    rot_true = y_true[..., 3:]
    rot_pred = y_pred[..., 3:]

    return tfg.quaternion.relative_angle(rot_true, rot_pred) * tf.constant(180. / math.pi)


def positional_diff_metric(y_true, y_pred):
    """Gives positional difference in millimeters."""
    pos_true = y_true[..., :3]
    pos_pred = y_pred[..., :3]

    return tf.linalg.norm(pos_true - pos_pred, axis=-1) * 1000


def pretrain_on_object_pose(pretrainable_component: tf.keras.Model,
                            epochs: int,
                            n_samples: int,
                            n_cameras=1,
                            load_data: bool = False,
                            name="visual_op",
                            dataset: Tuple[np.ndarray, np.ndarray] = None,
                            load_from: str = None):
    """Pretrain a visual component on prediction of cube position."""
    data_path = f"storage/data/pretraining/pose_data_{n_samples}_{n_cameras}c.tfrecord"
    if not load_data:
        dataset = gen_cube_quats_prediction_data(
            n_samples,
            data_path,
            n_cameras=n_cameras
        )
    else:
        dataset = load_dataset(data_path)

    # dataset = dataset.repeat(100000).shuffle(10000)
    # dataset = dataset.map(lambda x, y: (x, y))

    n_testset = 10000
    n_valset = 5000

    testset = dataset.take(n_testset)
    trainset = dataset.skip(n_testset)
    valset, trainset = trainset.take(n_valset), trainset.skip(n_valset)

    trainset = trainset.batch(128, drop_remainder=True)
    trainset = trainset.prefetch(AUTOTUNE)

    valset = valset.batch(128, drop_remainder=True)
    valset = valset.prefetch(AUTOTUNE)

    testset = testset.batch(128, drop_remainder=True)
    testset = testset.prefetch(AUTOTUNE)

    if load_from is None:
        model = pretrainable_component
        model(tf.expand_dims(next(iter(dataset))[0], 0))

        # chunk = list(tfds.as_numpy(dataset.take(8000).map(lambda x, y: y)))
        # chunk_mean = np.mean(chunk, axis=0)
        # output_layer = model.get_layer("output")
        # output_weights = output_layer.get_weights()
        # output_weights[1] = chunk_mean
        # output_layer.set_weights(output_weights)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer, loss="mse", metrics=[rotational_diff_metric, positional_diff_metric])

        # train and save encoder
        model.fit(x=trainset,
                  epochs=epochs,
                  validation_data=valset,
                  callbacks=[
                      tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
                  ],
                  shuffle=True)
        pretrainable_component.save(PRETRAINED_COMPONENTS_PATH + f"/{name}")
    else:
        print("Loading model...")
        model = tf.keras.models.load_model(load_from)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss="mse", metrics=[rotational_diff_metric, positional_diff_metric])
        print("Model loaded successfully.")

    train_mean = np.mean(list(tfds.as_numpy(trainset.unbatch().take(5000).map(lambda x, y: y))), axis=0)
    test_numpy = np.stack(list(tfds.as_numpy(testset.unbatch().map(lambda x, y: y))))
    print(f"This model achieves {model.evaluate(testset)}")
    print(f"A mean model would achieve {np.mean((test_numpy - train_mean) ** 2)}")


if __name__ == "__main__":

    tf.get_logger().setLevel('INFO')
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain a visual component on classification or reconstruction.")

    # general parameters
    parser.add_argument("task", nargs="?", type=str, choices=["classify", "reconstruct", "hand", "object",
                                                              "c", "r", "h", "o", "hp", "op"], default="o")
    parser.add_argument("--name", type=str, default="visual_component",
                        help="Name the pretraining to uniquely identify it.")
    parser.add_argument("--load", type=str, default=None, help=f"load the weights from checkpoint path")
    parser.add_argument("--epochs", type=int, default=30, help=f"number of pretraining epochs")

    # read arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # parameters
    n_cameras = 3
    n_samples = 1000000

    visual_component = OpenAIEncoder(shape=(VISION_WH, VISION_WH, 3), name=args.name, n_cameras=3)
    # visual_component = keras_cortex.cornet.cornet_z.PoseCORNetZ(7, name=args.name)

    os.makedirs(PRETRAINED_COMPONENTS_PATH, exist_ok=True)

    args.name = args.name + "_" + args.task[0]

    pretrain_on_object_pose(
        visual_component,
        epochs=args.epochs,
        n_samples=n_samples,
        n_cameras=n_cameras,
        load_data=False,
        name=args.name,
        load_from=args.load,
    )
