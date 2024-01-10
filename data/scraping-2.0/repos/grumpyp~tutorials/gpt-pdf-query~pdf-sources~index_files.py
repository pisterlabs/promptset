from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from typing import List
from langchain.schema import Document
import os
import pinecone
import pdf_loader


os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment="us-west1-gcp-free"  # next to api key in console
)


index_name = "langchain-demo"
active_indexes = pinecone.list_indexes()
if index_name not in active_indexes:
    pinecone.create_index(index_name, dimension=1536)


class Genie:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = TextLoader(self.file_path)
        self.documents = self.loader.load()
        self.texts = self.text_split(self.documents)
        self.vectordb = self.embeddings(self.texts)
        self.genie = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=self.vectordb.as_retriever())

    @staticmethod
    def text_split(documents: TextLoader):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    @staticmethod
    def embeddings(texts: List[Document]):
        embeddings = OpenAIEmbeddings()
        vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)
        return vectordb

    def ask(self, query: str):
        return self.genie.run(query)


if __name__ == "__main__":
    pdf_loader.load_pdf("misc/TM6.pdf")
    genie = Genie("result.txt")
    print(genie.ask("What happens if the power plug is broken? Can I fix it myself? I have experience with the thermomix!"))