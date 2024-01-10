import sys
import time

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

from kel.config import get_configs as config
from kel.constants.constants import emoji_time, emoji_money, emoji_info
from kel.utils.utils import copy_to_clipboard, print_in_color, before_ask_gpt_display, after_ask_gpt_display


def ask_ollama(company=None, question=None, model=None, prompt=None, max_tokens=None):
    """
    Call the Ollama API
    Returns:

    """

    before_ask_gpt_display(company=company, model=model)

    try:
        start_time = time.time()
        llm = Ollama(
            model=model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        response_time = time.time() - start_time
        if question is not None:
            question = str(question).strip() + str(prompt).strip()

        # Printing the response
        llm(question)
        after_ask_gpt_display(response_time=response_time)

        # TODO: Implement this
        # after_ask_gpt_display(consumed_tokens=model)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


