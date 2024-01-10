# -*- coding: utf-8 -*-

"""A module for Transformer-based VAE layers"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, Union

import tensorflow as tf
from tensorflow.keras import layers as tfl, backend as keras_backend
from tensorflow.keras.initializers import TruncatedNormal
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_tf_bert import TFBertEncoder
from transformers.models.openai.configuration_openai import OpenAIGPTConfig
from transformers.models.openai.modeling_tf_openai import TFBlock
from transformers.modeling_tf_utils import keras_serializable

from ae_sentence_embeddings.argument_handling import PositionalEmbeddingArgs, RegularizedEmbeddingArgs


@keras_serializable  # Note that this decorator adds the `get_config` method
class AeTransformerEncoder(TFBertEncoder):
    """Make TFBertEncoder serializable"""
    config_class = BertConfig


@keras_serializable
class AeTransformerDecoder(tfl.Layer):
    """Define a GPT decoder for an autoencoder"""
    config_class = OpenAIGPTConfig

    def __init__(self, config: OpenAIGPTConfig, **kwargs) -> None:
        """Layer initializer

        Args:
            config: GPT configuration object. `config.n_layer` contains information
                    about the number of Transformer decoder layers
            **kwargs: Keyword arguments for the superclass initializer
        """
        super().__init__(**kwargs)
        self.hidden = [TFBlock(config, scale=True, name=f"decoder_hidden_._{i}") for i in range(config.n_layer)]

    def call(
            self,
            inputs: Tuple[tf.Tensor, Optional[tf.Tensor]],
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Call the layer. Itt will not be able to output attentions or use head masks

        Args:
            inputs: Two tensors, the input hidden state and the attention mask.
                The latter can be `None`, which makes sense as the decoder performs causal attention masking
            training: Specifies whether the model is being used in training mode

        Returns:
            The output hidden state
        """
        hidden_state, attn_mask = inputs
        for transformer_block in self.hidden:
            # noinspection PyCallingNonCallable
            hidden_state = transformer_block(hidden_state, attn_mask, head_mask=None,
                                             output_attentions=False, training=training)[0]
        return hidden_state


class VaeSampling(tfl.Layer):
    """VAE sampling layer as suggested by A. Gérond in his book
    \"Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow\" (second edition, 2019), p. 588
    """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Sample from a Gaussian"""
        mean, log_var = inputs
        # Output type is correct below
        # noinspection PyTypeChecker
        return keras_backend.random_normal(tf.shape(log_var)) * keras_backend.exp(log_var / 2) + mean


class PostPoolingLayer(tfl.Layer):
    """A layer applied after pooling from the encoder to obtain Gaussian mean and variance values for a VAE"""

    def __init__(
            self,
            hidden_size: int,
            initializer_range: float = 0.02,
            **kwargs
    ) -> None:
        """Initialize the layer.

        Args:
            hidden_size: Hidden size of both input and output.
            initializer_range: stddev of a `TruncatedNormal` kernel initializer. Defaults to `0.02`.
            **kwargs: Parent class keyword arguments.
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        dense_params = {
            "units": hidden_size,
            "input_shape": (None, hidden_size),
            "kernel_initializer": TruncatedNormal(stddev=initializer_range)
        }
        self.post_pool_mean_dense = tfl.Dense(**dense_params, name="post_pool_mean_dense")
        self.post_pool_logvar_dense = tfl.Dense(**dense_params, name="post_pool_logvar_dense")

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Call the layer.

        Args:
            inputs: A tensor of shape `(batch_size, hidden_size)`

        Returns:
            Two tensors of shape `(batch_size, hidden_size)`
        """
        mean_tensor = self.post_pool_mean_dense(inputs)
        logvar_tensor = self.post_pool_logvar_dense(inputs)
        return mean_tensor, logvar_tensor

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        }


class RegularizedEmbedding(tfl.Layer):
    """An embedding layer regularized with layer normalization and dropout"""

    def __init__(self, embedding_args: Union[RegularizedEmbeddingArgs, PositionalEmbeddingArgs],
                 **kwargs) -> None:
        """Initialize the layer

        Args:
            embedding_args: embedding and regularization arguments
            **kwargs: Keyword arguments for the superclass initializer
        """
        super().__init__(**kwargs)
        self._vocab_size = embedding_args.vocab_size
        self._hidden_size = embedding_args.hidden_size
        self._layer_norm_eps = embedding_args.layer_norm_eps
        self._initializer_range = embedding_args.initializer_range
        self._hidden_dropout_prob = embedding_args.hidden_dropout_prob
        self.layernorm = tfl.LayerNormalization(epsilon=self._layer_norm_eps)
        self.dropout = tfl.Dropout(self._hidden_dropout_prob)
        self.static_embedding_layer = tfl.Embedding(
            input_dim=self._vocab_size,
            output_dim=self._hidden_size,
            embeddings_initializer=TruncatedNormal(stddev=self._initializer_range)
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Call the layer

        Args:
            inputs: An input ID tensor of shape `(batch_size, sequence_length)`
            training: Specify whether the model is being used in training mode

        Returns:
            An embedding tensor of shape `(batch_size, sequence_length, hidden_size)`
        """
        embeddings = self.static_embedding_layer(inputs)
        return self.dropout(self.layernorm(embeddings), training=training)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
            "hidden_dropout_prob": self.hidden_dropout_prob
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        embedding_args = PositionalEmbeddingArgs.collect_from_dict(config)
        additional_args = {k: v for k, v in config.items() if k not in embedding_args.to_dict().keys()}
        return cls(embedding_args, **additional_args)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def hidden_dropout_prob(self) -> float:
        return self._hidden_dropout_prob

    @property
    def initializer_range(self) -> float:
        return self._initializer_range

    @property
    def layer_norm_eps(self) -> float:
        return self._layer_norm_eps


class SinusoidalEmbedding(RegularizedEmbedding):
    """An implementation of the sinusoidal positional embeddings used in the original Transformer"""

    def __init__(self, embedding_args: PositionalEmbeddingArgs, **kwargs) -> None:
        """Initialize the layer

        Args:
            embedding_args: positional embedding arguments
            **kwargs: Keyword arguments for the superclass initializer
        """
        regularized_embedding_args = RegularizedEmbeddingArgs.collect_from_dict(
            embedding_args.to_dict())
        super().__init__(regularized_embedding_args, **kwargs)
        self._max_position_embeddings = embedding_args.max_position_embeddings
        self._min_freq = embedding_args.min_freq
        self.positional_matrix = self._get_positional_matrix()

    def _get_positional_matrix(self) -> tf.Tensor:
        """Get the full positional embedding matrix
        Original source of code:
        https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
        """
        with tf.name_scope("positional_matrix") as name:
            position = tf.range(self._max_position_embeddings, dtype=tf.float32)
            mask = tf.range(self._hidden_size)
            sin_mask = tf.cast(mask % 2, tf.float32)
            cos_mask = 1 - sin_mask
            exponent = 2 * (mask // 2)
            exponent = tf.cast(exponent, tf.float32) / tf.cast(self._hidden_size, tf.float32)
            freqs = self._min_freq ** exponent
            angles = tf.einsum('i,j->ij', position, freqs)
            return tf.add(tf.math.cos(angles) * cos_mask, tf.math.sin(angles) * sin_mask, name=name)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Call the layer

        Args:
            inputs: An input ID tensor of shape `(batch_size, sequence_length)`
            training: Specify whether the model is being used in training mode

        Returns:
            An embedding tensor of shape `(batch_size, sequence_length, hidden_size)`
        """
        sequence_length = tf.shape(inputs)[1]
        token_embeddings = self.static_embedding_layer(inputs)
        positional_encodings = self.positional_matrix[:sequence_length, :]
        embeddings = tf.add(token_embeddings, positional_encodings)
        return self.dropout(self.layernorm(embeddings), training=training)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "max_position_embeddings": self.max_position_embeddings,
            "min_freq": self.min_freq,
        }

    @property
    def max_position_embeddings(self) -> int:
        return self._max_position_embeddings

    @property
    def min_freq(self) -> float:
        return self._min_freq


class TrainablePositionalEmbedding(tfl.Layer):
    """An implementation of an embedding layer
    with trainable positional weights.
    """

    def __init__(self, embedding_args: PositionalEmbeddingArgs, **kwargs) -> None:
        """Initialize the layer."""
        super().__init__(**kwargs)
        self._layer_norm_eps = embedding_args.layer_norm_eps
        self._hidden_dropout_prob = embedding_args.hidden_dropout_prob
        self._vocab_size = embedding_args.vocab_size
        self._hidden_size = embedding_args.hidden_size
        self._initializer_range = embedding_args.initializer_range
        self._max_position_embeddings = embedding_args.max_position_embeddings

        self._layernorm = tfl.LayerNormalization(epsilon=embedding_args.layer_norm_eps)
        self._dropout = tfl.Dropout(embedding_args.hidden_dropout_prob)
        self._static_embedding_layer = tfl.Embedding(
            input_dim=embedding_args.vocab_size,
            output_dim=embedding_args.hidden_size,
            embeddings_initializer=TruncatedNormal(stddev=embedding_args.initializer_range)
        )
        self._pos_embedding_layer = tfl.Embedding(
            input_dim=embedding_args.max_position_embeddings,
            output_dim=embedding_args.hidden_size,
            embeddings_initializer=TruncatedNormal(stddev=embedding_args.initializer_range)
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Call the layer.

        Args:
            inputs: An input ID tensor of shape `(batch_size, sequence_length)`.
            training: Specify whether the model is being used in training mode.

        Returns:
            An embedding tensor of shape `(batch_size, sequence_length, hidden_size)`.
        """
        sequence_length = tf.shape(inputs)[1]
        pos_indices = tf.range(sequence_length)
        token_embeddings = self._static_embedding_layer(inputs)
        positional_encodings = self._pos_embedding_layer(pos_indices)
        embeddings = tf.add(token_embeddings, positional_encodings)
        return self._dropout(self._layernorm(embeddings), training=training)

    def get_config(self) -> Dict[str, Any]:
        """Get a config dictionary for serialization."""
        base_config = super().get_config()
        return {
            **base_config,
            "vocab_size": self._vocab_size,
            "hidden_size": self._hidden_size,
            "layer_norm_eps": self._layer_norm_eps,
            "initializer_range": self._initializer_range,
            "hidden_dropout_prob": self._hidden_dropout_prob,
            "max_position_embeddings": self._max_position_embeddings
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> TrainablePositionalEmbedding:
        """Initialize from a config dictionary."""
        embedding_args = PositionalEmbeddingArgs.collect_from_dict(config)
        additional_args = {k: v for k, v in config.items() if k not in embedding_args.to_dict().keys()}
        return cls(embedding_args, **additional_args)
