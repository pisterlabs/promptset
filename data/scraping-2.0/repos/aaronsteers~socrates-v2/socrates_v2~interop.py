"""Generic code for interop with Streamlit and Langchain."""

from langchain.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator

class LangChainStreamingCallbackToStreamlit(BaseCallbackHandler):
    """LangChain callback handler to stream responses from the chatbot.
    
    This gives a real-time 'typing' effect as text (tokens) are
    returned from the LLM, and is specific to rendering within Streamlit
    methods.
    """

    def __init__(self, container: DeltaGenerator, initial_text: str=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        """This is called to add text characters to the response."""
        self.text += token
        self.container.markdown(self.text)

