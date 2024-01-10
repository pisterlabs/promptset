from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
"""
https://python.langchain.com/en/latest/modules/indexes/getting_started.html
A lot of the magic is being hid in this VectorstoreIndexCreator. What is this doing?
There are three main steps going on after the documents are loaded:
    1.Splitting documents into chunks
    2.Creating embeddings for each document
    3.Storing documents and embeddings in a vectorstore
    
What are embeddings? 
embeddings measure the relatedness of text strings.
An embedding is a vector (list) of floating point numbers. 
The distance between two vectors measures their relatedness. 
Small distances suggest high relatedness and large distances suggest low relatedness.
"""

loader = TextLoader('../test.txt')
documents = loader.load()

# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# select which embeddings we want to use.
embeddings = OpenAIEmbeddings()
# create the vectorstore to use as the index.
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
retriever = db.as_retriever()

# create a chain and use it to answer questions!
llm = OpenAI()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)


# index_creator = VectorstoreIndexCreator(
#     vectorstore_cls=Chroma,
#     embedding=OpenAIEmbeddings(),
#     text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# )
