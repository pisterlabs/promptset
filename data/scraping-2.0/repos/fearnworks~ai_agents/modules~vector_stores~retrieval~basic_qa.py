from modules.llm.defaults import get_default_cloud_chat_llm
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.vectorstores.base import VectorStore
from dataclasses import dataclass

@dataclass
class QAgentConfig:
    index: VectorStore
    qa: BaseRetrievalQA

class QAgent:
    def __init__(self, index: VectorStore, qa: BaseRetrievalQA):
        self.index = index
        self.llm = get_default_cloud_chat_llm()
        self.qa_chain = qa

    def ask(self, question: str):
        resp = self.qa_chain(question)
        self.process_llm_response(resp)
        return resp
    
    ## Cite sources
    def process_llm_response(self, llm_response):
        print(llm_response['result'])
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])


def get_default_qa(index: VectorStore) -> QAgent:
    llm = get_default_cloud_chat_llm()
    qa_chain = RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=index.as_retriever(), return_source_documents=True)
    qagent = QAgent(index, qa_chain)
    return qagent