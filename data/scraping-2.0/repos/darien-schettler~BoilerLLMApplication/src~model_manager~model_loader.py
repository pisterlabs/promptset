from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import BaseCallbackHandler, StreamingStdOutCallbackHandler


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text, unsafe_allow_html=True)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


def get_openai_chat_model(model_name="gpt-3.5-turbo-0613", temperature=0.7, use_streaming=False,
                          streaming_cb="streamlit", st_container=None, verbose=True, **kwargs):
    """
    ["gpt-3.5-turbo-16", "gpt-3.5-turbo-0613"]
    ["streamlit", "stdout"]
    """
    streaming_cb = None
    if streaming_cb == "stdout" and use_streaming:
        streaming_cb = [StreamingStdOutCallbackHandler()]
    elif streaming_cb == "streamlit" and use_streaming:
        streaming_cb = [StreamlitCallbackHandler(st_container)]

    chat_model = ChatOpenAI(
        model_name=model_name, temperature=temperature, verbose=verbose, streaming=use_streaming,
        callbacks=streaming_cb, **kwargs
    )

    return chat_model
