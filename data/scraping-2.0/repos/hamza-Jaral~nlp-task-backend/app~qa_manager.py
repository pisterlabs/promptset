from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import JSONLoader
from langchain.chains.summarize import load_summarize_chain


class QAManager:

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.prompt = hub.pull("rlm/rag-prompt")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(),
                                collection_name="chroma_db",)
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        self.rag_chain = ({"context": self.retriever | self.format_docs,
                           "question": RunnablePassthrough()}
                          | self.prompt
                          | self.llm
                          | StrOutputParser()
                          )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, query):
        return self.rag_chain.invoke(query)


class DataIndexer:

    def _metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["document"] = record.get("doc_name")
        metadata["page"] = record.get("pagenum")

        return metadata

    def create_chroma_db_embedding(self, file_paths):

        files_data = []
        for filepath in file_paths:
            loader = JSONLoader(
                file_path=filepath,
                jq_schema='.document[]',
                content_key="text",
                metadata_func=self._metadata_func
            )

            data = loader.load()
            files_data.extend(data)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )

        all_splits = text_splitter.split_documents(files_data)

        emb = OpenAIEmbeddings()
        Chroma.from_documents(
            collection_name="chroma_db",
            documents=all_splits,
            embedding=emb,
            persist_directory="./chroma_db"
        )


class Summarizer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.chain = load_summarize_chain(self.llm, chain_type="stuff")

    def load_docs(self, document):
        pass
