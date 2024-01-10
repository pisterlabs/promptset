from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_lit import LlmEmbedding


class ConversationalRetrievalChainAgent:
    def __init__(self, llm: BaseLanguageModel, retriever: BaseRetriever):
        question_prompt = PromptTemplate.from_template(
            """You are QA Bot. If you don't know the answer, just say that you don't know, don't try to make up an answer.""")
        self.llm_qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            condense_question_prompt=question_prompt,
            return_source_documents=True,
            verbose=False
        )

    def ask(self, question: str):
        history = []
        result = self.llm_qa({"question": question, "chat_history": history})
        history = [(question, result["answer"])]
        return result["answer"]


class RetrievalQAAgent:
    """
    db.get_store() + RetrievalQAAgent
    不知道為什麼找不到 document
    """
    def __init__(self, llm:  BaseLanguageModel, retriever: BaseRetriever):
        chain_type_kwargs = {
            "verbose": False,
            "prompt": self.create_prompt(),
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"
            )
        }
        self.llm_qa = RetrievalQA.from_chain_type(llm=llm,
                                                  chain_type="stuff",
                                                  verbose=False,
                                                  retriever=retriever,
                                                  chain_type_kwargs=chain_type_kwargs)

    def ask(self, question: str):
        result = self.llm_qa({"query": question})
        return result["answer"]

    def create_prompt(self, prompt_template: str = None):
        if prompt_template is None:
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            You are QA Bot. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Only return the helpful answer below and nothing else.
            Helpful answer:"""
        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )


def create_parent_document_retriever(vector_store: VectorStore):
    store = InMemoryStore()
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    big_chunks_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return big_chunks_retriever


class FaissRetrieval:
    def __init__(self, llm_embedding: LlmEmbedding):
        self.llm_embedding = llm_embedding

    def get_retriever(self, docs):
        vector_store = FAISS.from_documents(docs, self.llm_embedding.embedding)
        return create_parent_document_retriever(vector_store)


class Retrieval:
    def __init__(self, vector_db, llm, llm_embedding: LlmEmbedding):
        self.vector_db = vector_db
        self.llm = llm
        self.llm_embedding = llm_embedding

    def get_retriever(self, collection_name: str):
        store = self.vector_db.get_store(collection_name)
        return store.as_retriever(search_kwargs={"k": 5})

    def get_parent_document_retriever(self, collection_name: str):
        vector_store = self.vector_db.get_store(collection_name)
        store = InMemoryStore()
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        big_chunks_retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        return big_chunks_retriever

    def add_parent_document(self, collection_name: str, docs: list[Document]):
        big_chunks_retriever = self.get_parent_document_retriever(collection_name)
        big_chunks_retriever.add_documents(docs)
