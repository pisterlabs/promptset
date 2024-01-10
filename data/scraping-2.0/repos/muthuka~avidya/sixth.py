from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import os

llm = OpenAI()


# load document
loader = PyPDFLoader("./docs/harrypotter1.pdf")
documents = loader.load()

# chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce")

# query = "What is Diagon Alley?"
# chain.run(input_documents=documents, question=query)

# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# select which embeddings we want to use
embeddings = OpenAIEmbeddings()

# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)

# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# create a chain to answer questions
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

query = "what is Diagon Alley?"
result = qa({"query": query})
print(result["result"])
