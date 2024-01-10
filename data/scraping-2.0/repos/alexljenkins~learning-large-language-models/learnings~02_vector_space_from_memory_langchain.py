"""
Uses Langchain to:
1. Load and split document text into chunks
2. create a vector space from the embeddings by sending them to OpenAI's API (storing the results in memory).
3. create a QA system that retrieves semantically similar chunks from the vector store and sends them as context with the question to OpenAI's chatbot
"""

from dotenv import load_dotenv

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()
embeddings = OpenAIEmbeddings()
loader = DirectoryLoader('datasets/rick_data', glob="txt/*.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=docsearch.as_retriever()
)

def query(q):
    print("Question: ", q)
    print("Answer: ", qa.run(q))

query("who made butter robot?")
query("what is butter robot's only purpose?")
query("Who does Morty love more than anyone else?")
