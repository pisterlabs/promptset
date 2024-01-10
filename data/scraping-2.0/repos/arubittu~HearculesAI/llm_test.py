# langchain based llm with prompt template, chat history, tools, RAG, streaming
# V1 - chat_history, grimorie, streaming 
import os
import time
from threading import Thread
from queue import Queue, Empty
from threading import Thread
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections.abc import Generator
from langchain.llms import OpenAI, OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler


# api key
os.environ["OPENAI_API_KEY"] = 'key'

# Defined a QueueCallback, which takes as a Queue object during initialization. Each new token is pushed to the queue.
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()


# Create a function that will return our generator
def langchain_stream(input_text) -> Generator:

    # Create a Queue
    q = Queue()
    job_done = object()

    # Initialize the LLM we'll be using
    prompt ='''you are an receptionist at taj hotel mumbai.
                user question : {input}
                response:
                    '''
    template = PromptTemplate(template=prompt,input_variables=['input'])
    
    llm = ChatOpenAI(
        model ='gpt-3.5-turbo',
        streaming=True, 
        callbacks=[QueueCallback(q)], 
        temperature=0.1
    )
    chain = LLMChain(llm=llm,prompt=template)
    # Create a funciton to call - this will run in a thread
    def task():
        resp = chain.run(input_text)
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token
        except Empty:
            continue

if __name__ == "__main__":
    s = time.time()
    for next_token in langchain_stream("joke??"):
        t = time.time()
        print(t-s)
        print(next_token)
        #print(content)
        
    