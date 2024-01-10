from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_handlers = [
    StreamingStdOutCallbackHandler()
]

llm_ollama = Ollama(
    model="mistral",
    num_gpu=120,
    temperature=0,
    verbose=False,
    callback_manager=CallbackManager(callback_handlers)
)