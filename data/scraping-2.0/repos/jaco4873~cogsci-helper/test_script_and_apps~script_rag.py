"""
Built into app:
https://docs.chainlit.io/integrations/langchain
"""
from operator import itemgetter

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

from prompts import answer_prompt
from retriever import retriever

# GPT4 SWITCH
USE_GPT4 = False  # set to true if relevant
MODEL_NAME = "gpt-4-turbo" if USE_GPT4 else "gpt-3.5-turbo"

# set retriever, prompt and model
model = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.2,  # precise: the model always return the same response to identical prompts
    callbacks=[
        StreamingStdOutCallbackHandler()
    ],  # allows for streaming if LLM supports it
    streaming=True,  # allows for streaming if LLM supports it
)


runnable = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | answer_prompt
    | model
    | StrOutputParser()
)


runnable.invoke({"question": "What is Cognitive Neuroscience?"})
