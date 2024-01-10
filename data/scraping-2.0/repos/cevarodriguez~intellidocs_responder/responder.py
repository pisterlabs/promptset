from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

import os

# OpenAI platform key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Load pdf file and split into chunks
loader = PyPDFLoader("sample_data/Photosynthesis.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages = loader.load_and_split(text_splitter)

# Prepare vector store
directory = 'index_store'
vector_index = Chroma.from_documents(pages, OpenAIEmbeddings(), persist_directory=directory)
vector_index.persist() # actually the Chroma client automatically persists the indexes when it is disposed - however, better safe than sorry :-)

# Prepare the retriever chain
retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k":6})
qa_interface = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

# Test query (Optional)
print("Test query: ", qa_interface("What is aerobic respiration? Return 3 paragraphs and a headline as markdown."))

#Adding additional docs
loader = PyPDFLoader("sample_data/graphite.pdf")
pages_new = loader.load_and_split(text_splitter)
_ = vector_index.add_documents(pages_new)
vector_index.persist()

#Adding memory
conv_interface = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), retriever=retriever)
chat_history = []
query = "What is photosyntheses?"

# First chat query
result = conv_interface({"question": query, "chat_history": chat_history})

print("Photosynthesis meaning: ", result["answer"])

# Second query, using the previous queries as memory

# Add previous conversation to chat history
chat_history.append((query, result["answer"]))

# Shorten the last sentence
query = "Can you shorten this sentence please?"
result = conv_interface({"question": query, "chat_history": chat_history})

print("Shortened answer: ", result["answer"])