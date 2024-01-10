from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings

# @see https://python.langchain.com/docs/integrations/llms/ollama
# setup:
# ./ollama serve
# ./ollama run llama2
# run: python ollama-query.py 


llm = Ollama(
   model="llama2:13b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
llm("Tell me about the history of Napoleon")


