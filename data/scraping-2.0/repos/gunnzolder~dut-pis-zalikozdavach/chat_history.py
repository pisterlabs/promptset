# from langchain.llms import Ollama
# ollama = Ollama(base_url='http://localhost:11434', model='mistral')



from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 

from langchain.chains import ConversationChain  
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOllama


chat = ChatOllama(base_url='http://localhost:11434', model='mistral',
             callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

# messages = [
#     SystemMessage(content="You are a helpful assistant that translates English to French."),
#     HumanMessage(content="I love programming.")
# ]
# print(chat(messages))

# ollama = Ollama(base_url='http://localhost:11434', model='mistral',
#              callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

# print(ollama('why is the sky blue?'))



conversation = ConversationChain(llm=chat)  
conversation.run("Translate this sentence from English to French: I love programming.")

conversation.run("Translate it to German.")