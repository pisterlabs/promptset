# this is the file that we're going to run any time that we want to ask a question & grab some content out of our vector database (refer to the fact-QA.png for demo)

from langchain.embeddings import OpenAIEmbeddings # embeddings
from langchain.vectorstores.chroma import Chroma # for storing embeddings
from dotenv import load_dotenv # for loading variables from .env
from langchain.chat_models import ChatOpenAI # model
from langchain.chains import RetrievalQA # chain
from redundant_filter_retriever import RedundantFilterRetriever # custom retriever to find relevant docs & remove any duplicate automatically
import langchain

langchain.debug = True # for printing all intermediate outputs and inputs on the chain


# ------ MODEL ------
# load variables from .env:
load_dotenv()
chat = ChatOpenAI()


# ------ EMBEDDINGS ------
# initialize an Embedding API Object (a gateway for accessing OpenAI's embedding generation capabilities):
embeddings = OpenAIEmbeddings()


# ------ STORE EMBEDDING (for the user's question of text strings) ------
# (database creation) create a Chroma database instance:
db = Chroma(
    embedding_function=embeddings, # specify the func used to generate embeddings from the user's question
    persist_directory="emb" # specify the directory to store the calculated embedding for future use
)


# ------ RETRIEVAL CHAIN ------
# # (from the db instance) create a retriever object to retrieve info from the db:
# retriever = db.as_retriever()

# or use custom retriever instead:
retriever = RedundantFilterRetriever(
    # provide the 2 customization attributes:
    embeddings=embeddings,
    chroma=db
)

# create a retrieval-based question-answering chain:
chain = RetrievalQA.from_chain_type(
    chain_type="stuff", # specify the type of chain to create (take some context from the vector store and "stuff" it into the prompt)
    retriever=retriever, # the retriever object for retrieving info from db
    llm=chat, # model used to generate answers
    # verbose=True
)


# ------ RESULT ------
result = chain.run("What is an interesting fact about the English language?")
print(result)
