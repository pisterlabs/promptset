import inspect
from typing import Any
from typing import Dict
from typing import List
from unittest.mock import Mock
from unittest.mock import patch

import openai
import pytest

import ai_ghostfunctions.ghostfunctions
from ai_ghostfunctions import ghostfunction
from ai_ghostfunctions.types import Message


def test_aicallable_function_decorator_has_same_signature() -> None:
    def generate_n_random_words(n: int, startswith: str) -> List[str]:  # type: ignore[empty-body]
        """Return a list of `n` random words that start with `startswith`."""
        pass

    with patch.object(ai_ghostfunctions.ghostfunctions.os, "environ"):  # type: ignore[attr-defined]
        decorated_function = ghostfunction(generate_n_random_words)
        assert inspect.signature(decorated_function) == inspect.signature(
            generate_n_random_words
        )


def test_aicallable_function_decorator() -> None:
    expected_result = ["returned value from openai"]
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ) as patched:

        @ghostfunction
        def generate_n_random_words(n: int, startswith: str) -> List[str]:  # type: ignore[empty-body]
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")
        patched.assert_called_once()

    assert result == expected_result


def test_aicallable_function_decorator_with_open_close_parens() -> None:
    expected_result = ["returned value from openai"]
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ) as patched:

        @ghostfunction()
        def generate_n_random_words(n: int, startswith: str) -> List[str]:  # type: ignore[empty-body]
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")
        patched.assert_called_once()

    assert result == expected_result


def test_aicallable_function_decorator_with_custom_prompt_function() -> None:
    new_prompt = [Message(role="user", content="this is a new prompt")]

    expected_result = ["returned value from openai"]
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ) as patched:

        @ghostfunction(prompt_function=lambda f, **kwargs: new_prompt)
        def generate_n_random_words(n: int, startswith: str) -> List[str]:  # type: ignore[empty-body]
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")
        patched.assert_called_once()
    mock_callable.assert_called_once_with(messages=new_prompt)

    assert result == expected_result


@pytest.mark.parametrize(
    "expected_result,annotation",
    [
        ("return a string", str),
        (b"return bytes", bytes),
        (1.23, float),
        (11, int),
        (("return", "tuple"), tuple),
        (["return", "list"], List[str]),
        ({"return": "dict"}, Dict[str, str]),
        ({"return", "set"}, set),
        (True, bool),
        (None, None),
    ],
)
def test_ghostfunction_decorator_returns_expected_type(
    expected_result: Any, annotation: Any
) -> None:
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ) as patched:

        @ghostfunction
        def generate_n_random_words(n: int, startswith: str) -> annotation:
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")
        patched.assert_called_once()

    assert result == expected_result


def test_ghostfunction_decorator_with_custom_agg_function() -> None:
    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {
                "choices": [
                    {"message": {"content": "good"}},
                    {"message": {"content": "goose"}},
                    {"message": {"content": "goodness"}},
                ]
            }
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ) as patched:

        @ghostfunction(aggregation_function=lambda choices: ",".join(choices))
        def generate_n_random_words(n: int, startswith: str) -> str:  # type: ignore[empty-body]
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(n=5, startswith="goo")
        patched.assert_called_once()

    assert result == "good,goose,goodness"


@pytest.mark.parametrize(
    "expected_result,annotation",
    [
        ("return a string", str),
    ],
)
def test_ghostfunction_can_be_called_with_positional_arguments(
    expected_result: Any, annotation: Any
) -> None:
    mock_return_result = str(expected_result)

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": mock_return_result}}]}
        )
    )
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ):

        @ghostfunction
        def generate_n_random_words(n: int, startswith: str) -> annotation:
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(5, "goo")
        result2 = generate_n_random_words(5, startswith="goo")

    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ):

        @ghostfunction()
        def generate_n_random_words(n: int, startswith: str) -> annotation:
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(5, "goo")
        result2 = generate_n_random_words(5, startswith="goo")

    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ):

        @ghostfunction(ai_callable=mock_callable)
        def generate_n_random_words(n: int, startswith: str) -> annotation:
            """Return a list of `n` random words that start with `startswith`."""
            pass

        result = generate_n_random_words(5, "goo")
        result2 = generate_n_random_words(5, startswith="goo")

    assert result == result2 == expected_result


def test_ghostfunction_decorator_errors_if_no_return_type_annotation() -> None:
    expected_result = "returned value from openai"

    mock_callable = Mock(
        return_value=openai.openai_object.OpenAIObject.construct_from(
            {"choices": [{"message": {"content": expected_result}}]}
        )
    )

    # test with bare ghostfunction
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ):
        with pytest.raises(ValueError):

            @ghostfunction
            def f(a: int):  # type: ignore[no-untyped-def]
                """This is an example that doesn't have a return annotation."""
                pass

    # test with ghostfunction with open-close parens
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=mock_callable,
    ):
        with pytest.raises(ValueError):

            @ghostfunction()
            def f2(a: int):  # type: ignore[no-untyped-def]
                """This is an example that doesn't have a return annotation."""
                pass


def test_ghostfunction_decorator_errors_if_no_docstring() -> None:
    with patch.object(
        ai_ghostfunctions.ghostfunctions,
        "_default_ai_callable",
        return_value=Mock(),
    ):
        with pytest.raises(ValueError):

            @ghostfunction
            def f(a: int):  # type: ignore[no-untyped-def]
                pass

        with pytest.raises(ValueError):

            @ghostfunction()
            def f2(a: int):  # type: ignore[no-untyped-def]
                pass


def toy_function(x: str) -> str:  # type: ignore[empty-body]
    """This is.

    multiline
    docstring
    """


def test__make_chatgpt_message_from_function_works_well_with_multiline_docstrings() -> (
    None
):
    msg = ai_ghostfunctions.ghostfunctions._make_chatgpt_message_from_function(
        toy_function, x="this"
    )
    assert msg["content"] == (
        "from mymodule import toy_function\n"
        "\n"
        "# The return type annotation for the function toy_function is <class 'str'>\n"
        "# The docstring for the function toy_function is the following:\n"
        "# This is.\n"
        "# \n"
        "#     multiline\n"
        "#     docstring\n"
        "#     \n"
        "result = toy_function(x='this')\n"
        "print(result)\n"
    )


@pytest.mark.parametrize(
    "ai_result,expected_return_type,expected_function_result",
    [
        ("a bare string", str, "a bare string"),
        ('"a double quoted string"', str, "a double quoted string"),
        ("'a single quoted string'", str, "a single quoted string"),
    ],
)
def test___parse_ai_result(
    ai_result: str, expected_return_type: Any, expected_function_result: Any
) -> None:
    ai_result_wrapper = {"choices": [{"message": {"content": ai_result}}]}
    assert (
        ai_ghostfunctions.ghostfunctions._parse_ai_result(
            ai_result_wrapper,
            expected_return_type,
        )
        == expected_function_result
    )


def test___parse_ai_result_non_default_agg_function() -> None:
    ai_result_wrapper = {
        "choices": [
            {"message": {"content": "c1"}},
            {"message": {"content": "c2"}},
            {"message": {"content": "c3"}},
        ]
    }
    assert (
        ai_ghostfunctions.ghostfunctions._parse_ai_result(
            ai_result_wrapper, str, aggregation_function=lambda x: x[1]
        )
        == "c2"
    )
    assert (
        ai_ghostfunctions.ghostfunctions._parse_ai_result(
            ai_result_wrapper, str, aggregation_function=",".join
        )
        == "c1,c2,c3"
    )
