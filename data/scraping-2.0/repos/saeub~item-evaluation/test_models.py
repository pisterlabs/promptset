import openai
from openai.openai_object import OpenAIObject
import pytest

from data import Answer, Item
from models import OpenAIModel


@pytest.fixture
def mock_openai_tf(monkeypatch):
    openai.api_key = "test"

    def mock_completion(*args, **kwargs):
        return OpenAIObject.construct_from(
            {
                "choices": [
                    {
                        "text": "\nR.",
                        "logprobs": {
                            "top_logprobs": [
                                {
                                    "\n": -0.000001,
                                    "R": -0.000002,
                                    "F": -0.000003,
                                    " ": -0.000004,
                                    "r": -0.000005,
                                },
                                {
                                    "R": -0.000001,
                                    "F": -0.000002,
                                    " ": -0.000003,
                                    "x": -0.000004,
                                    "r": -0.000005,
                                },
                            ],
                        },
                    },
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                },
            }
        )

    def mock_chat_completion(*args, **kwargs):
        return OpenAIObject.construct_from(
            {
                "choices": [{"message": {"content": "\nR."}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                },
            }
        )

    monkeypatch.setattr("openai.Completion.create", mock_completion)
    monkeypatch.setattr("openai.ChatCompletion.create", mock_chat_completion)


@pytest.fixture
def mock_openai_pick(monkeypatch):
    openai.api_key = "test"

    def mock_completion(*args, **kwargs):
        return OpenAIObject.construct_from(
            {
                "choices": [
                    {
                        "text": "\nA)",
                        "logprobs": {
                            "top_logprobs": [
                                {
                                    "\n": -0.000001,
                                    "A": -0.000002,
                                    "B": -0.000003,
                                    " ": -0.000004,
                                    "r": -0.000005,
                                },
                                {
                                    "A": -0.000001,
                                    "B": -0.000002,
                                    " ": -0.000003,
                                    "C": -0.000004,
                                    "D": -0.000005,
                                },
                            ],
                        },
                    },
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                },
            }
        )

    def mock_chat_completion(*args, **kwargs):
        return OpenAIObject.construct_from(
            {
                "choices": [{"message": {"content": "\nA)"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                },
            }
        )

    monkeypatch.setattr("openai.Completion.create", mock_completion)
    monkeypatch.setattr("openai.ChatCompletion.create", mock_chat_completion)


@pytest.mark.parametrize(
    ("lang", "chat"),
    [
        ("de", False),
        ("de", True),
    ],
)
def test_openai_model_tf(mock_openai_tf, lang, chat):
    model = OpenAIModel(
        "gpt-whatever", lang=lang, chat=chat, true_false=True, multi=True
    )
    predictions = model.answer(
        "Text",
        Item(
            question="Question",
            answers=[
                Answer(text="Text", correct=True),
                Answer(text="Text", correct=False),
            ],
            multiple=True,
        ),
    )
    assert len(predictions) == 2
    assert predictions[0].pred_correct is True
    assert predictions[0].model_output == "\nR."
    assert predictions[1].pred_correct is True
    assert predictions[1].model_output == "\nR."


@pytest.mark.parametrize(
    ("lang", "chat", "multi"),
    [
        ("de", False, False),
        ("de", True, False),
        ("en", False, False),
        ("en", True, False),
        ("de", False, True),
        ("de", True, True),
        ("en", False, True),
        ("en", True, True),
    ],
)
def test_openai_model_pick(mock_openai_pick, lang, chat, multi):
    model = OpenAIModel(
        "gpt-whatever", lang=lang, chat=chat, true_false=False, multi=multi
    )
    predictions = model.answer(
        "Text",
        Item(
            question="Question",
            answers=[
                Answer(text="Text", correct=True),
                Answer(text="Text", correct=False),
            ],
            multiple=False,
        ),
    )
    assert len(predictions) == 2
    assert predictions[0].pred_correct is True
    assert predictions[0].model_output == "\nA)"
    assert predictions[1].pred_correct is False
    assert predictions[1].model_output == "\nA)"
