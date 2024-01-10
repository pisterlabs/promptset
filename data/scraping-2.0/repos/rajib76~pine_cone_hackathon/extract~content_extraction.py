import os

import pinecone
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Pinecone

from parsers.word_parser import WordParser
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

class FRDExtract():
    def __init__(self,
                 parser:WordParser(),
                 emb:OpenAIEmbeddings()
                 ):
        self.module = "FRD"
        self.parser = parser
        self.emb=emb

    def load_doc(self,dir):
        documents = self.parser.load_docs()
        return documents

    def split_doc(self,docs):
        text_splitter = TokenTextSplitter(
            chunk_size=100,
            chunk_overlap=0
        )
        for doc in docs:
            chunks = text_splitter.split_text(doc.page_content)
            source = doc.metadata["source"]
            chunk_docs = [Document(page_content=text,metadata={"text":text,"source":source}) for text in chunks]
            for chunk_doc in chunk_docs:
                print(chunk_doc)
            return chunk_docs

    def load_pine_index(self,docs,index_name = "frd"):
        # initialize pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_ENV,  # next to api key in console
        )

        docsearch = Pinecone.from_documents(docs, self.emb, index_name=index_name)

        return docsearch

if __name__=="__main__":
    parser = WordParser()
    emb=OpenAIEmbeddings()
    frd=FRDExtract(parser=parser,emb=emb)
    docs = frd.load_doc("../data/FRD.docx")
    # print(docs)
    chunks = frd.split_doc(docs)
    for chunk in chunks:
        print(chunk)
    docsearch = frd.load_pine_index(chunks)
    print(docsearch)
