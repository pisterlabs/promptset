import os
import re
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )
    
def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

os.environ["OPENAI_API_KEY"] = "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"

embeddings = OpenAIEmbeddings()
vectorstore = Chroma("langchain_store", embeddings, persist_directory="./data/CHROMA_DB_2")
vectorstore.persist()

if is_docker():
    docs_dir = "./data/documentation"
else: 
    docs_dir = "./documentation"
    
# Read all documents including subdirectories
for root, dirs, files in os.walk(docs_dir):
    for file in files:
        if file.endswith(".xml") or file.endswith(".md") or file.endswith(".html") or file.endswith(".yaml"):
            # Load the document, split it into chunks, embed each chunk and load it into the vector store.
            raw_document = TextLoader(os.path.join(root, file)).load()
            # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=0,
                                               separators=[" ", ",", "\n"])

            documents = text_splitter.split_documents(raw_document)
            if len(documents) > 0 and file.endswith('.html') > 0:
                for document in documents:
                    document.page_content = clean_html(document.page_content)


            vectorstore.add_documents(documents)
vectorstore.persist()
print(vectorstore.similarity_search("What is Fogify?"))


