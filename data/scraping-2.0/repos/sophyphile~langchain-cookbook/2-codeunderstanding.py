# Code Understanding
# Co-pilot-esque functionality that can help answer questions from a specific library, and help generate new code

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Helper to read local files
import os

# Vector support
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Model and chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Text splitters
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=openai_api_key)

embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=openai_api_key)

# Put a small Python package The Fuzz in the data folder of this repo
# The loop below will go through each file in the library and load it up as a doc
root_dir = 'data/thefuzz'
docs = []

# Go through each folder
for dirpath, dirnames, filenames in os.walk(root_dir):

    # Go through each file
    for file in filenames:
        try:
            # Load up the file as a doc and split
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

print (f"You have {len(docs)} documents\n")
print ("------ Start Document ------")
print (docs[0].page_content[:300])

# Embed and store them in a docstore. This will make an API call to OpenAI.
docsearch = FAISS.from_documents(docs, embeddings)

# Get our retriever ready
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

query = "What function do I use if I want to find the most similar item in a list of items?"
output = qa.run(query)
print(output)

query = "Can you write the code to use the process.extractOne() function? Only respond with code. No other text or explanation"
output = qa.run(query)
print(output)

