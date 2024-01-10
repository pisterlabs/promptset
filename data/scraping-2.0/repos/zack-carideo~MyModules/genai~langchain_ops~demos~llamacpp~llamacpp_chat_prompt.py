##DEMO of using llama2chat model with LLMChain
##Reference: https://python.langchain.com/docs/integrations/chat/llama2_chat

"""
All ChatModels implement the Runnable interface,
which comes with default implementations of all methods, 
ie. ainvoke, batch, abatch, stream, astream.
This gives all ChatModels basic support for async,
streaming and batch, which by default is implemented as below:

"""
from pathlib import Path
from os.path import expanduser

from langchain.llms import LlamaCpp
from langchain_experimental.chat_models import Llama2Chat
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

template_messages = [
    SystemMessage(content="You are a helpful assistant.")
    , MessagesPlaceholder(variable_name="chat_history")
    , HumanMessagePromptTemplate.from_template("{text}")
    ,
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)

filename = 'llama-2-7b-langchain-chat.Q4_K.gguf'
local_dir = "/home/zjc1002/Mounts/llms/llama-2-7b-langchain-chat-gguf"
modelpath = Path(local_dir, filename) 
model_path = expanduser(modelpath)

#create llamaccp instance
llm = LlamaCpp(
    model_path=model_path,
    streaming=False,
)

#wrap model into llama2chat instance 
model = Llama2Chat(llm=llm)

#use chat model together with prompt_template and converstaino memory in LLMChain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

#ask initial question
print(
    chain.run(
        
        text="What can I see in Vienna? Propose a few locations. Names only, no details."
    )
)

#based on the model response, ask a follow up question 
print(
    chain.run(
        text="tell me more about #2."
    )
)