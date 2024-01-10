# from langchain.llms import Ollama
# ollama = Ollama(base_url='http://localhost:11434', model='mistral')



from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler                                  
ollama = Ollama(base_url='http://localhost:11434', model='mistral',
             callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

print(ollama('why is the sky blue?'))



