import pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.vectorstores import Pinecone


def execute(language="pt-BR"):
    pinecone.init()

    file_name = f"../rules.{language}.md"
    loader = UnstructuredMarkdownLoader(file_name)
    documents = loader.load()
    markdown_splitter = MarkdownTextSplitter(chunk_size=750, chunk_overlap=0)
    docs = markdown_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    return Pinecone.from_documents(docs, embeddings, index_name="uno-rules", namespace=f"{language}_rules")
