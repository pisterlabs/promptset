from rubatochat.core.chat import ChatOpenAIChat
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
_OPENAI_API_KEY = "SECRET"

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
def test_base_chat():
    
    chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3
                      ,openai_api_key=_OPENAI_API_KEY)
    
    messages = [
        SystemMessage(content="You are an expert data scientist"),
        HumanMessage(content="Write a Python script that trains a neural network on simulated data ")
    ]
    
    response=chat(messages)
    print(response.content,end='\n')
    
    return None
    
def testChatOpenAIChat():
    
    callback = AsyncIteratorCallbackHandler()
    _chat = ChatOpenAIChat(
        streaming=True,
        openai_api_key=_OPENAI_API_KEY,
        verbose=True,
        callbacks=[callback]
        
    )
    print(_chat.question("hello, who are you?"))
    
    return _chat

#test_base_chat()    
_chat = testChatOpenAIChat()

_response = _chat.question("who is beethoven, introduce him in 300 words")
print(_response)

import pickle

_serialized=pickle.dumps(_chat) 
pickle.loads(_serialized)
#print(_serialized)

