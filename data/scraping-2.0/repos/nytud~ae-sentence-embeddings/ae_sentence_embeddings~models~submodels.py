# -*- coding: utf-8 -*-

"""A module for models that are intended to be used as layers in a more complex model during pre-training.
Defining them as a model allows to use them separately after pre-training
"""

# Note that does not often recognize whether that Keras layer or model is callable.
# This is the reason why the corresponding inspection were suppressed for some functions and classes.

from __future__ import annotations
from typing import Tuple, Optional, Literal, Dict, Any
from types import MappingProxyType
from copy import deepcopy

import tensorflow as tf
from tensorflow.keras import Model as KModel
from tensorflow.keras import layers as tfl
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.utils import register_keras_serializable
from transformers import TFSharedEmbeddings
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.openai.configuration_openai import OpenAIGPTConfig
from transformers.modeling_tf_utils import keras_serializable

from ae_sentence_embeddings.layers import (
    AeTransformerEncoder,
    AeTransformerDecoder,
    AveragePoolingLayer,
    PMeansPooling,
    CLSPlusSEPPooling,
    AeGRUDecoder,
    AeTransformerGRUDecoder,
    TrainablePositionalEmbedding
)
from ae_sentence_embeddings.regularizers import KLDivergenceRegularizer
from ae_sentence_embeddings.modeling_tools import process_attention_mask, make_decoder_inputs
from ae_sentence_embeddings.argument_handling import (
    RnnLayerArgs, RnnArgs,
    PositionalEmbeddingArgs,
    KlArgs
)


@register_keras_serializable(package="ae_sentence_embeddings.models")
class SentAeEncoder(KModel):
    """The full encoder part of an AE"""

    def __init__(self, config: BertConfig,
                 pooling_type: Literal["average", "cls_sep", "p_means"],
                 **kwargs) -> None:
        """Layer initializer.

        Args:
            config: A BERT configuration object.
            pooling_type: Pooling method, `'average'`, `'cls_sep'`
                or `'p_means'`.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(**kwargs)
        self._transformer_config = config
        self._pooling_type = pooling_type.lower()
        if self._pooling_type == "average":
            self._pooling = AveragePoolingLayer()
        elif self._pooling_type == "cls_sep":
            self._pooling = CLSPlusSEPPooling()
        elif self._pooling_type == "p_means":
            self._pooling = PMeansPooling()
        else:
            raise NotImplementedError(f"Unknown pooling type: {pooling_type}")
        self._embedding_layer = TrainablePositionalEmbedding(
            PositionalEmbeddingArgs.collect_from_dict(config.to_dict()))
        self._transformer_encoder = AeTransformerEncoder(self._transformer_config)

    # noinspection PyCallingNonCallable
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
        """Call the encoder.

        Args:
            inputs: Input ID tensor with shape `(batch_size, sequence_length)`
                    and attention mask with shape `(batch_size, sequence_length)`.
            training: Specifies whether the model is being used in training mode.

        Returns:
            A pooled tensor and the Transformer encoder outputs.

        """
        input_ids, attention_mask = inputs
        embeddings = self._embedding_layer(input_ids, training=training)
        mod_attention_mask = process_attention_mask(attention_mask, embedding_dtype=embeddings.dtype)
        encoder_outputs = self._transformer_encoder(
            hidden_states=embeddings,
            attention_mask=mod_attention_mask,
            head_mask=[None] * self._transformer_config.num_hidden_layers,
            past_key_values=[None] * self._transformer_config.num_hidden_layers,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            training=training
        )
        sequence_output = encoder_outputs[0]
        pooling_result = self._pooling((sequence_output, attention_mask))
        return pooling_result, encoder_outputs + (embeddings,)

    @property
    def pooling_type(self) -> Literal["average", "cls_sep", "p_means"]:
        # noinspection PyTypeChecker
        return self._pooling_type  # The return type is correct, PyCharm may complain because of the `str.lower` call

    def get_config(self) -> Dict[str, Any]:
        return {
            "encoder_config": self._transformer_config.to_dict(),
            "pooling_type": self._pooling_type
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> SentAeEncoder:
        encoder_config = BertConfig(**config.pop("encoder_config"))
        return cls(encoder_config, **config, **kwargs)

    @property
    def transformer_config(self) -> MappingProxyType:
        return MappingProxyType(self._transformer_config.to_dict())


@register_keras_serializable(package="ae_sentence_embeddings.models")
class SentVaeEncoder(SentAeEncoder):
    """The full encoder part of a VAE"""
    _keras_serializable = True

    def __init__(
            self,
            config: BertConfig,
            pooling_type: Literal["average", "cls_sep", "p_means"] = "average",
            reg_args: Optional[KlArgs] = None,
            **kwargs
    ) -> None:
        """Initialize the encoder.

        Args:
            config: A BERT configuration object.
            pooling_type: Pooling type, `'average'`, `'cls_sep'` or `'p_means'`.
                Defaults to `'average'`.
            reg_args: KL loss regularization arguments. Optional.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(config, pooling_type, **kwargs)
        hidden_size = 2 * config.hidden_size if self._pooling_type in {"cls_sep", "p_means"} \
            else config.hidden_size
        if reg_args is None:
            reg_args = KlArgs(iters=0, warmup_iters=1)
        reg_args = reg_args.to_dict()
        self._post_pooling = tfl.Dense(
            units=hidden_size * 2,
            input_shape=(None, hidden_size),
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            activity_regularizer=KLDivergenceRegularizer(**reg_args),
            name="post_pooling_dense"
        )
        # Define a layer to split the Gaussian vectors to mean and logvar
        self._post_pooling_splitter = tfl.Lambda(lambda x: tf.split(x, 2, axis=-1))
        self._reg_args = reg_args
        self._reg_args["iters"] = 0

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, Tuple[tf.Tensor, ...]]:
        """Call the encoder.

        Args:
            inputs: Input IDs tensor with shape `(batch_size, sequence_length)`
                and attention mask with shape `(batch_size, sequence_length)`.
            training: Specifies whether the model is being used in training mode.

        Returns:
            Two pooled tensors (mean and log variance for VAE sampling) and the Transformer encoder outputs.
        """
        pooling_output, encoder_outputs = super().call(inputs, training=training)
        # noinspection PyCallingNonCallable
        post_pooling_tensor = self._post_pooling(pooling_output)
        mean, logvar = self._post_pooling_splitter(post_pooling_tensor)
        return mean, logvar, encoder_outputs

    @property
    def reg_args(self) -> MappingProxyType:
        """Get regularization arguments. The `iters` argument will be set to zero."""
        return MappingProxyType(self._reg_args)

    def get_config(self) -> Dict[str, Any]:
        """Get serialization configuration.
        Note that the regularization in a serialized model has no effect.
        This is due to the fact that the `iters` argument of the KL regularizer
        will be set to the constant `0`.
        """
        base_config = super(SentVaeEncoder, self).get_config()
        return {
            **base_config,
            "reg_args": deepcopy(self._reg_args)
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> SentAeEncoder:
        """Initialize the model from a configuration object."""
        reg_args = KlArgs(**config.pop("reg_args"))
        encoder_config = BertConfig(**config.pop("encoder_config"))
        return cls(config=encoder_config, reg_args=reg_args, **config, **kwargs)


# noinspection PyAbstractClass
@keras_serializable  # Note that this decorator adds the `get_config` method
class SentAeDecoder(KModel):
    """The full decoder part of the autoencoder"""
    config_class = OpenAIGPTConfig

    def __init__(self, config: OpenAIGPTConfig, **kwargs) -> None:
        """Layer initializer

        Args:
            config: A BERT configuration object
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(**kwargs)
        self._transformer_config = config
        self._transformer_decoder = AeTransformerDecoder(self._transformer_config)
        embedding_args = PositionalEmbeddingArgs(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.n_positions,
            hidden_size=config.n_embd,
            hidden_dropout_prob=config.embd_pdrop,
            layer_norm_eps=config.layer_norm_epsilon,
            initializer_range=config.initializer_range
        )
        self._embedding_layer = TrainablePositionalEmbedding(embedding_args)
        self._out_dense = tfl.Dense(config.vocab_size,
                                    kernel_initializer=TruncatedNormal(stddev=config.initializer_range))

    # noinspection PyCallingNonCallable
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the decoder.

        Args:
            inputs: An input embedding tensor of shape `(batch_size, hidden_size)`.
                    an input token ID tensor of shape `(batch_size, sequence_length, hidden_size)` and
                    an attention mask tensor of shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in training mode

        Returns:
            Logits for next token prediction
        """
        sent_embeddings, input_ids, attention_mask = inputs
        token_embeddings = self._embedding_layer(input_ids[:, 1:], training=training)
        attention_mask = process_attention_mask(
            attention_mask=make_decoder_inputs(attention_mask),
            embedding_dtype=token_embeddings.dtype
        )
        encodings = tf.concat([tf.expand_dims(sent_embeddings, axis=1), token_embeddings], axis=1)
        hidden_output = self._transformer_decoder((encodings, attention_mask), training=training)
        logits = self._out_dense(hidden_output)
        return logits


class SentAeGRUDecoder(KModel):
    """A GRU-based full decoder."""

    def __init__(self, config: RnnArgs, **kwargs) -> None:
        """Layer initializer.

        Args:
            config: An RNN configuration dataclass object.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(**kwargs)
        self._gru_config = config
        config_dict = config.to_dict()
        vocab_size = config_dict.pop("vocab_size")
        init_range = config_dict.pop("initializer_dev")
        self._decoder_embedding = TFSharedEmbeddings(
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=init_range,
            name="decoder_embedding"
        )
        self._decoder = AeGRUDecoder(**config_dict)

    # noinspection PyCallingNonCallable
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the decoder

        Args:
            inputs: An sentence embedding tensor of shape `(batch_size, hidden_size)` and
                an input token ID tensor of shape `(batch_size, sequence_length)`.
            training: Specifies whether the model is being used in training mode.

        Returns:
            Logits for next token prediction
        """
        sent_embeddings, token_ids = inputs
        token_embeddings = self._decoder_embedding(token_ids, mode="embedding")
        hidden_output = self._decoder((sent_embeddings, token_embeddings), training=training)
        logits = self._decoder_embedding(hidden_output, mode="linear")
        return logits

    def get_config(self) -> Dict[str, Any]:
        return {"decoder_config": self._gru_config.to_dict()}

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> SentAeGRUDecoder:
        decoder_config = RnnArgs(**config.pop("decoder_config"))
        return cls(decoder_config, **kwargs)

    @property
    def transformer_config(self) -> MappingProxyType:
        return MappingProxyType(self._gru_config.to_dict())


def parallel_decoders(
        sent_hidden_size: int,
        tok_hidden_size: int,
        vocab_size: int,
        linear_stddev: float,
        decoder_class: type,
        decoder_kwargs: Dict[str, Any],
        name: Optional[str] = None,
) -> KModel:
    """Define parallel decoders with the functional API

    Args:
        sent_hidden_size: Hidden size of the input sentence representations.
        tok_hidden_size: Hidden size of the input token representations.
        vocab_size: Vocabulary size for the output logits.
        linear_stddev: `TruncatedNormal` stddev argument of the final dense layer that outputs logits.
        decoder_class: Decoder class to use.
        name: Optional. Model name as a string.
        decoder_kwargs: Keyword arguments to initialize the decoder instances

    Returns:
        A functional Keras model
    """
    sent_embeddings1 = tfl.Input(shape=(sent_hidden_size,), dtype=tf.float32)
    token_embeddings1 = tfl.Input(shape=(None, tok_hidden_size), dtype=tf.float32)
    sent_embeddings2 = tfl.Input(shape=(sent_hidden_size,), dtype=tf.float32)
    token_embeddings2 = tfl.Input(shape=(None, tok_hidden_size), dtype=tf.float32)

    branch1_out = decoder_class(**decoder_kwargs)((sent_embeddings1, token_embeddings1))
    branch2_out = decoder_class(**decoder_kwargs)((sent_embeddings2, token_embeddings2))
    outputs = tfl.concatenate([branch1_out, branch2_out], axis=0)
    outputs = tfl.Dense(
        vocab_size,
        kernel_initializer=TruncatedNormal(stddev=linear_stddev),
    )(outputs)
    return KModel(inputs=[sent_embeddings1, token_embeddings1, sent_embeddings2, token_embeddings2],
                  outputs=outputs, name=name)
