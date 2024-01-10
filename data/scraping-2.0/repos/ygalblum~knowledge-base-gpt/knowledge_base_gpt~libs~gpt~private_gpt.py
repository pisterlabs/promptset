import os

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from knowledge_base_gpt.libs.common import constants

model = os.environ.get("MODEL", "llama2-uncensored")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
ollama_host = os.environ.get("OLLAMA_HOST", 'localhost')

better_prompt_template = """
Use the following pieces of context to answer the question at the end. You can do two things:
 1. Return the answer to a question ONLY if the context contains the answer to the question.
 2. Return ONLY the text 'I dont know' if the context does not contain the answer to a question.
{context}


Question: {question}
Helpful Answer:
"""

PROMPT = PromptTemplate(
    template=better_prompt_template, input_variables=["context", "question"]
)

class PrivateGPT():
    __instance = None

    @staticmethod
    def get_instance():
        if PrivateGPT.__instance is None:
            PrivateGPT()
        return PrivateGPT.__instance

    def __init__(self):
        if PrivateGPT.__instance is not None:
            raise Exception("This class is a singleton!")
        self._qa = None
        self._hide_source = True
        PrivateGPT.__instance = self

    def initialize(self, hide_source=True, verbose=False):
        self._hide_source = hide_source
        embeddings = HuggingFaceEmbeddings(model_name=constants.embeddings_model_name)
        db = Chroma(persist_directory=constants.persist_directory, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        llm = Ollama(model=model, base_url=f"http://{ollama_host}:11434")
        self._qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=not self._hide_source,
            chain_type_kwargs={"prompt":PROMPT, "verbose":verbose}
        )

    def answer_query(self, query):
        res = self._qa(query)
        answer, _ = res['result'], [] if self._hide_source else res['source_documents']
        return answer
