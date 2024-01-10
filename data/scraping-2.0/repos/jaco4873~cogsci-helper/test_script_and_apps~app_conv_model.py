"""
Built into app:
https://docs.chainlit.io/integrations/langchain
"""
from operator import itemgetter

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.messages import get_buffer_string
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

from doc_combiner import _combine_documents
from prompts import answer_prompt, condense_prompt
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


# Feed first call to LLM the chat-history
_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | condense_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


conversational_qa_chain = _inputs | _context | answer_prompt | model

conversational_qa_chain.invoke(
    {
        "question": "What is the cerebral Cortex?",
        "chat_history": [],
    }
)
