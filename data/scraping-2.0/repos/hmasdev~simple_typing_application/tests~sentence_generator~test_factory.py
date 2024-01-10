import pytest

from simple_typing_application.const.sentence_generator import ESentenceGeneratorType  # noqa
from simple_typing_application.models.config_models.sentence_generator_config_model import (  # noqa
    OpenAISentenceGeneratorConfigModel,
    HuggingfaceSentenceGeneratorConfigModel,
    StaticSentenceGeneratorConfigModel,
)
from simple_typing_application.sentence_generator.factory import (
    create_sentence_generator,
    _select_class_and_config_model,
)
from simple_typing_application.sentence_generator.huggingface_sentence_generator import HuggingfaceSentenceGenerator  # noqa
from simple_typing_application.sentence_generator.openai_sentence_generator import OpenaiSentenceGenerator  # noqa
from simple_typing_application.sentence_generator.static_sentence_generator import StaticSentenceGenerator  # noqa


@pytest.mark.parametrize(
    "sentence_generator_type, expected_class, expected_config_model",
    [
        (
            ESentenceGeneratorType.OPENAI,
            OpenaiSentenceGenerator,
            OpenAISentenceGeneratorConfigModel,
        ),
        (
            ESentenceGeneratorType.HUGGINGFACE,
            HuggingfaceSentenceGenerator,
            HuggingfaceSentenceGeneratorConfigModel,
        ),
        (
            ESentenceGeneratorType.STATIC,
            StaticSentenceGenerator,
            StaticSentenceGeneratorConfigModel,
        ),
    ]
)
def test_select_class_and_config_model(
    sentence_generator_type: ESentenceGeneratorType,
    expected_class: type,
    expected_config_model: type,
):

    # execute
    sentence_generator_cls, sentence_generator_config_model = _select_class_and_config_model(sentence_generator_type)  # noqa

    # assert
    assert sentence_generator_cls is expected_class
    assert sentence_generator_config_model is expected_config_model


def test_select_class_and_config_model_raise_value_error():
    # execute
    with pytest.raises(ValueError):
        _select_class_and_config_model('invalid_key_monitor_type')  # type: ignore  # noqa


@pytest.mark.parametrize(
    "sentence_generator_type, sentence_generator_config_dict, expected_class",
    [
        (
            ESentenceGeneratorType.OPENAI,
            OpenAISentenceGeneratorConfigModel().model_dump(),
            OpenaiSentenceGenerator,
        ),
        (
            ESentenceGeneratorType.HUGGINGFACE,
            HuggingfaceSentenceGeneratorConfigModel().model_dump(),
            HuggingfaceSentenceGenerator,
        ),
        (
            ESentenceGeneratorType.STATIC,
            StaticSentenceGeneratorConfigModel(text_kana_map={}).model_dump(),
            StaticSentenceGenerator,
        ),
        (
            ESentenceGeneratorType.OPENAI,
            {},
            OpenaiSentenceGenerator,
        ),
        (
            ESentenceGeneratorType.HUGGINGFACE,
            {},
            HuggingfaceSentenceGenerator,
        ),
        (
            ESentenceGeneratorType.STATIC,
            {},
            StaticSentenceGenerator,
        ),
    ]
)
def test_create_sentence_generator(
    sentence_generator_type: ESentenceGeneratorType,
    sentence_generator_config_dict: dict[str, str | float | int | bool | None | dict | list],  # noqa
    expected_class: type,
    mocker,
):

    # mock
    # for OpenaiSentenceGenerator
    mocker.patch('simple_typing_application.sentence_generator.openai_sentence_generator.ChatOpenAI')  # noqa
    mocker.patch('simple_typing_application.sentence_generator.openai_sentence_generator.ConversationChain')  # noqa
    mocker.patch('simple_typing_application.sentence_generator.openai_sentence_generator.ConversationBufferMemory')  # noqa
    # for HuggingfaceSentenceGenerator
    mocker.patch('simple_typing_application.sentence_generator.huggingface_sentence_generator.AutoModelForCausalLM.from_pretrained')  # noqa
    mocker.patch('simple_typing_application.sentence_generator.huggingface_sentence_generator.AutoTokenizer.from_pretrained')  # noqa
    mocker.patch('simple_typing_application.sentence_generator.huggingface_sentence_generator.pipeline')  # noqa
    # for StaticSentenceGenerator
    # None

    # execute
    sentence_generator = create_sentence_generator(
        sentence_generator_type,
        sentence_generator_config_dict,
    )

    # assert
    assert isinstance(sentence_generator, expected_class)


def test_create_sentence_generator_raise_import_error(mocker):

    # mock
    mocker.patch(
        'simple_typing_application.sentence_generator.factory._select_class_and_config_model',  # noqa
        side_effect=NameError,
    )

    # execute
    with pytest.raises(ImportError):
        create_sentence_generator(
            ESentenceGeneratorType.HUGGINGFACE,
            {},
        )


def test_create_sentence_generator_raise_value_error(mocker):

    # mock
    mocker.patch(
        'simple_typing_application.sentence_generator.factory._select_class_and_config_model',  # noqa
        side_effect=ValueError,
    )

    # execute
    with pytest.raises(ValueError):
        create_sentence_generator(
            'invalid_sentence_generator_type',  # type: ignore
            {},
        )
