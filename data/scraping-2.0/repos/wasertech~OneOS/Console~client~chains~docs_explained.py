# Documentation Explainer

# You have a question about some piece of code?
# Ain't nobody got time to read the docs?
# Just ask this chain about it.
# It can also extrapolate custom code from the docs.

import subprocess

from langchain.document_loaders import readthedocs
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

from client.chains.vectorstores import get_vectorstore_from_readthedocs
from client.chains.embeddings import get_embeddings
from client.chains.models import get_llm


db_path = "docs_db"

class DocExplainer:

    def __init__(
            self,
            readthedocs_url,
            name="Expert",
            max_tokens=2048,
            temperature=0.0,
            streaming=False,
            callbacks=[],
            verbose=False,
        ):
        self.name = name
        self.readthedocs_url = readthedocs_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.streaming = streaming
        self.callbacks = callbacks
        self.verbose = verbose
        
        self._initialize_llm()
        self._initialize_vectorstore()
        self._initialize_chain()
    
    def _initialize_llm(self):
        # TODO: Try multiple LLMs and pick the first one that answered.
        self.llm = get_llm(max_tokens=self.max_tokens, temperature=self.temperature, streaming=self.streaming, callbacks=self.callbacks)
        self.embeddings = get_embeddings()

    def _initialize_vectorstore(self):
        self.vectorstore = get_vectorstore_from_readthedocs(self.readthedocs_url, self.embeddings)
    
    def _initialize_chain(self):
        self.question_generator = LLMChain(
            llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT, # callback_manager=manager
        )
        self.doc_chain = load_qa_chain(
            self.llm, chain_type="stuff", prompt=QA_PROMPT, # callback_manager=manager
        )

        self.memory = ConversationBufferMemory(memory_key="chat_history")

        self.qa = ConversationalRetrievalChain(
            retriever=self.vectorstore.as_retriever(),
            combine_docs_chain=self.doc_chain,
            question_generator=self.question_generator,
            #callback_manager=manager,
            memory=self.memory,
            response_if_no_docs_found="I don't know.",
        )
    
    def __call__(self, question):
        return self.qa.run(question=question)

if __name__ == "__main__":
    expert = DocExplainer("https://python.langchain.com/docs/")
    q = (
    "Write a custom multi-step agent class "
    "that uses a tool retriever "
    "to make the prompt shorter "
    "and more relevant "
    "relative to the query's context."
    "\n"
    )
    print(expert(q))
