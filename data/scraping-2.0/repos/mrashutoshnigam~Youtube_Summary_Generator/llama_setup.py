from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

llm = LlamaCpp(
    model_path="model/ggml-model-q4_0.gguf"
temperature = 0.0,
top_p = 1,
n_ctx = 6000,
callback_manager = callback_manager,
verbose = True)
