import logging
from logging import getLogger, Logger

from .base import BaseSentenceGenerator  # noqa
try:
    from .huggingface_sentence_generator import HuggingfaceSentenceGenerator  # noqa
except ImportError:
    logging.warning(
        'Failed to import HuggingfaceSentenceGenerator. '
        'If you want to use HuggingfaceSentenceGenerator, `pip install simple_typing_application[huggingface]`.'  # noqa
    )
from .openai_sentence_generator import OpenaiSentenceGenerator  # noqa
from .static_sentence_generator import StaticSentenceGenerator  # noqa
from ..const.sentence_generator import ESentenceGeneratorType  # noqa
from ..models.config_models.sentence_generator_config_model import (  # noqa
    BaseSentenceGeneratorConfigModel,
    OpenAISentenceGeneratorConfigModel,
    HuggingfaceSentenceGeneratorConfigModel,
    StaticSentenceGeneratorConfigModel,
)


def _select_class_and_config_model(sentence_generator_type: ESentenceGeneratorType) -> tuple[type, type]:  # noqa

    if sentence_generator_type == ESentenceGeneratorType.OPENAI:
        return OpenaiSentenceGenerator, OpenAISentenceGeneratorConfigModel
    elif sentence_generator_type == ESentenceGeneratorType.HUGGINGFACE:
        return HuggingfaceSentenceGenerator, HuggingfaceSentenceGeneratorConfigModel  # noqa
    elif sentence_generator_type == ESentenceGeneratorType.STATIC:
        return StaticSentenceGenerator, StaticSentenceGeneratorConfigModel
    else:
        raise ValueError(f'Unsupported sentence generator type: {sentence_generator_type}')  # noqa


def create_sentence_generator(
    sentence_generator_type: ESentenceGeneratorType,
    dict_config: dict[str, str | float | int | bool | None | dict | list],
    logger: Logger = getLogger(__name__),
) -> BaseSentenceGenerator:

    # select sentence generator class and config model
    try:
        sentence_generator_cls, sentence_generator_config_model = _select_class_and_config_model(sentence_generator_type)  # noqa
    except NameError:
        raise ImportError(f'Failed to import sentence generator class and config model for sentence_generator_type={sentence_generator_type}')  # noqa

    # create sentence generator
    logger.debug(f'create {sentence_generator_cls.__name__}')
    sentence_generator_config: BaseSentenceGeneratorConfigModel = sentence_generator_config_model(**dict_config)  # noqa
    sentence_generator: BaseSentenceGenerator = sentence_generator_cls(**sentence_generator_config.model_dump())    # type: ignore # noqa

    return sentence_generator
