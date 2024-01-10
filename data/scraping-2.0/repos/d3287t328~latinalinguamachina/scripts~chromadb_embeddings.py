# allows you to chat with all the text or markdown files in the dir /tmp/chroma
import os
from glob import glob
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = ""

# UTF-8 Text Loader
class UTF8TextLoader(TextLoader):
    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [self.create_document(text, {"source": self.file_path})]

# Modified DirectoryLoader to include subdirectories
class MyDirectoryLoader(DirectoryLoader):
    def __init__(self, directory, glob="**/*", loader_cls=UTF8TextLoader, **loader_kwargs):
        super().__init__(directory, glob, loader_cls, **loader_kwargs)

# Load documents from directory
def load_documents_from_directory(directory, patterns):
    documents = []
    for pattern in patterns:
        loader = MyDirectoryLoader(directory, glob=pattern)
        documents.extend(loader.load())
    return documents

# Directory and patterns (include subdirectories)
directory_path = '/tmp/chroma/'
patterns = ["**/*.txt", "**/*.md"]

# Load and process documents
documents = load_documents_from_directory(directory_path, patterns)

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Embedding and persistence
persist_directory = 'db'
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
vectordb = None

# Load from disk
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Create retriever and QA chain
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

# LLM Response Processing
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# Interactive QA
while True:
    query = input("Enter your query: ")
    if query.lower() == 'quit':
        break
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

# Cleanup
vectordb.delete_collection()
vectordb.persist()
