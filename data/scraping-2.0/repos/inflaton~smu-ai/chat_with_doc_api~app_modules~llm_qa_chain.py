from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.vectorstores.base import VectorStore

from app_modules.llm_inference import LLMInference


class QAChain(LLMInference):
    vectorstore: VectorStore

    def __init__(self, vectorstore, llm_loader, doc_id_to_vectorstore_mapping=None):
        super().__init__(llm_loader)
        self.vectorstore = vectorstore
        self.doc_id_to_vectorstore_mapping = doc_id_to_vectorstore_mapping

    def get_chain(self, inputs) -> Chain:
        return self.create_chain(inputs)

    def create_chain(self, inputs) -> Chain:
        vectorstore = self.vectorstore
        if "chat_id" in inputs:
            if inputs["chat_id"] in self.doc_id_to_vectorstore_mapping:
                vectorstore = self.doc_id_to_vectorstore_mapping[inputs["chat_id"]]

        qa = ConversationalRetrievalChain.from_llm(
            self.llm_loader.llm,
            vectorstore.as_retriever(search_kwargs=self.llm_loader.search_kwargs),
            max_tokens_limit=self.llm_loader.max_tokens_limit,
            return_source_documents=True,
        )

        return qa
