import os

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

# take environment variable OPENAI_API_KEY
key = os.environ.get("OPENAI_API_KEY")
if not key:
    # or take key from file
    with open("OPENAI_API_KEY", "r") as f:
        key = f.read()

if not key:
    raise ValueError("Please add your OpenAI API key to the file OPENAI_API_KEY")

openai_chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=key,
)

openai_chat_stream = ChatOpenAI(
    callbacks=[StreamingStdOutCallbackHandler()],
    model_name="gpt-3.5-turbo",
    openai_api_key=key,
    streaming=True,
)
