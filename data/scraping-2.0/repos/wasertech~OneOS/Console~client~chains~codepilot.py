# Documentation Explainer

# You have a question about some piece of code?
# Ain't nobody got time to read the docs?
# Just ask this chain about it.
# It can also extrapolate custom code from the docs.

from langchain.document_loaders import readthedocs
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
#from langchain import OpenAI

from client.chains.vectorstores import get_vectorstore_from_readthedocs
from client.chains.embeddings import get_embeddings
from client.chains.models import get_llm

from rich.console import Console
from rich.markdown import Markdown

db_path = "docs_db"

class CodePilot:

    def __init__(
            self,
            #readthedocs_url,
            name="CodePilot",
            max_tokens=2048,
            temperature=0.0,
            streaming=False,
            callbacks=[],
            verbose=False,
        ):
        self.name = name
        #self.readthedocs_url = readthedocs_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.streaming = streaming
        self.callbacks = callbacks
        self.verbose = verbose
        
        self._initialize_llm()
        self._initialize_vectorstore()
        self._initialize_prompt()
        self._initialize_chain()
    
    def _initialize_llm(self):
        # TODO: Try multiple LLMs and pick the first one that answered.
        #self.llm = OpenAI(temperature=0)
        self.llm = get_llm(max_tokens=self.max_tokens, temperature=self.temperature, streaming=self.streaming, callbacks=self.callbacks)
        self.embeddings = get_embeddings()

    def _initialize_vectorstore(self):
        # Make sure you have ingested the docs into the vectorstore.
        # Use ./docs_explained.py to do that.
        self.vectorstore = get_vectorstore_from_readthedocs("", self.embeddings)
    
    def _initialize_prompt(self):
        prompt_template = """Use the context below to write a piece of code to satisfy the query below:
            Context: {context}
            Query: {query}
            Code: """
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

    def _initialize_chain(self):
        self.generator = LLMChain(
            llm=self.llm, prompt=self.prompt
        )
    
    def __call__(self, query):
        docs = self.vectorstore.similarity_search(query, k=4)
        docs_content = "\n".join([doc.page_content for doc in docs])
        input_dict = {"context": docs_content, "query": query}
        return self.clear_output(self.generator.run(input_dict))

    def clear_output(self, output):
        output = output.replace("<|im_end|>", "")
        output = output.replace("<|endoftext|>", "")
        return output

if __name__ == "__main__":
    console = Console()
    codepilot = CodePilot()
    q = (
    "Write a custom multi-step agent class "
    "that uses a tool retriever "
    "to make the prompt shorter "
    "and more relevant "
    "relative to the query's context."
    "\n"
    )
    code = codepilot(q)
    console.print(Markdown(code))
