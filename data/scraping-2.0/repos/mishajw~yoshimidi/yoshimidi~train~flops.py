from yoshimidi.train.model.transformer_config import TransformerConfig
from yoshimidi.train.training_config import TrainingConfig


def calculate_flops(
    transformer_config: TransformerConfig,
    training_config: TrainingConfig,
    time_per_batch_secs: float,
) -> float:
    """
    Approximates the number of floating point operations achieved per-second.
    """
    num_flop = calculate_num_flop(transformer_config)
    return (
        num_flop
        * 2
        * training_config.batch_size
        * transformer_config.context_window
        / time_per_batch_secs
    )


def calculate_num_flop(
    transformer_config: TransformerConfig,
) -> int:
    """
    Approximates the number of floating point operations performed to process a single
    token in a forward pass.

    Derived from OpenAI's scaling laws paper:
    https://arxiv.org/pdf/2001.08361.pdf#page=7
    """
    return (
        2 * calculate_num_parameters(transformer_config)
        + 2
        * transformer_config.num_layers
        * transformer_config.context_window
        * transformer_config.attention_head_size
    )


def calculate_num_parameters(config: TransformerConfig) -> int:
    """
    Approximates the number of parameters in the model.

    Derived from OpenAI's scaling laws paper:
    https://arxiv.org/pdf/2001.08361.pdf#page=7
    """
    return (
        2
        * config.residual_stream_size
        * config.num_layers
        * (2 * config.attention_head_size + config.feed_forward_size)
    )
