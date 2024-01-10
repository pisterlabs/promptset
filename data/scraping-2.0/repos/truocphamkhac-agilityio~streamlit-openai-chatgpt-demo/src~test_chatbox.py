import datetime
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice


# See https://github.com/openai/openai-python/issues/715#issuecomment-1809203346
def create_chat_completion(response: str, role: str = "assistant") -> ChatCompletion:
    return ChatCompletion(
        id="foo",
        model="gpt-3.5-turbo",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=response,
                    role=role,
                ),
            )
        ],
        created=int(datetime.datetime.now().timestamp()),
    )


@patch("openai.resources.chat.Completions.create")
@patch("streamlit.connection")
def test_validate_credentials(conn, openai_create):
    """Test the validate credentials script"""
    at = AppTest.from_file("validate_credentials.py")

    # Set up all the mocks
    at.secrets["OPENAI_API_KEY"] = "sk-..."
    openai_create.return_value = create_chat_completion("Streamlit is really awesome!")

    # Run the script and compare results
    at.run()
    print(at)
    assert at.markdown[0].value == "Streamlit is really awesome!"
    assert not at.exception
