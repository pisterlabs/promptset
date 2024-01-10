import logging
import os

import pinecone

# load environment variable
from dotenv import load_dotenv

# For streaming
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone

from .init_vector_db import initialize_vector_store

load_dotenv()

# Initialize all required env variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY"


class LLMCompletion:
    def __init__(
        self,
        # vectorstore: Pinecone,
        model_name="gpt-3.5-turbo-16k",
        callbacks=[StreamingStdOutCallbackHandler()],
        top_p=0.2,
    ):
        # self.vectorstore = vectorstore
        [self.vectorstore, self.index] = initialize_vector_store()
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=OPENAI_API_KEY,
            callbacks=callbacks,
            temperature=1.0,
            streaming=True,
            max_tokens=500,
            model_kwargs={"top_p": top_p},
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.vectorstore.as_retriever()
        )

        self.qa_with_source = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.vectorstore.as_retriever()
        )

    def complete(self, query):
        return self.qa.run(query)

    def complete_with_source(self, query):
        return self.qa_with_source(query)
