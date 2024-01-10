from langchain.llms import Ollama
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from nova_chat.constants import RemoteLLM
from langchain.callbacks.base import BaseCallbackHandler

class LLMFactory:
    @staticmethod
    def getChat(llm: RemoteLLM, st = None):
        if st:  # provide streamlit to set proper output streaming
            callback_manager = CallbackManager([StreamHandler(st)])
        else:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
        if "gpt" not in llm.value.model:
            return ChatOllama(
                base_url=llm.value.base_url, 
                model=llm.value.model, 
                callback_manager = callback_manager,
            )
        else:
            return ChatOpenAI(
                model_name=llm.value.model,
                temperature=0.1,
                openai_api_key=llm.value.open_api_key,
                callback_manager = callback_manager,
                streaming=True,
                verbose=True,
            )
        

class StreamHandler(BaseCallbackHandler):
    """Stream printing handler."""
    def __init__(self, st, initial_text=""):
        self.st = st
        self.text=initial_text
        with st.chat_message("assistant"):
            self.message_placeholder = st.empty()
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        try:
            self.message_placeholder.markdown(self.text)
        except:
            self.message_placeholder.write(self.text)
