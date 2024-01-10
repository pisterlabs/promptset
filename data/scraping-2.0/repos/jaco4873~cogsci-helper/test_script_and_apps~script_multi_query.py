'''
Built into app:
https://docs.chainlit.io/integrations/langchain
'''
from operator import itemgetter


from langchain.chat_models import ChatOpenAI
from langchain.llms import openai
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import StrOutputParser
import logging

from retriever import retriever
from prompts import answer_prompt

# GPT4 SWITCH
USE_GPT4 = False # set to true if relevant
model_name = "gpt-4-turbo" if USE_GPT4 else "gpt-3.5-turbo"

# set model
model = ChatOpenAI(
    model = model_name,
    temperature = 0.2, # precise: the model always return the same response to identical prompts
    callbacks = [StreamingStdOutCallbackHandler()], # allows for streaming if LLM supports it
    streaming = True # allows for streaming if LLM supports it
    )

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=ChatOpenAI()
)

# set logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# set runnable
runnable = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question") | retriever_from_llm,
    }
    | answer_prompt
    | model
    | StrOutputParser()
)

# invoke runnable
runnable.invoke({"question": 'What is Cognitive Neuroscience'})

