%pip install chromadb==0.3.21 tiktoken==0.3.3 sqlalchemy==2.0.15

### Add Credentials
# For many of the services that we'll using in the notebook, we'll need a HuggingFace API key so this cell will ask for it:
# HuggingFace Hub: https://huggingface.co/inference-api

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_1234567890987654321"

### Step 1 - Loading Documents into our Vector Store
# For this system we'll leverage the ChromaDB vector database (https://www.trychroma.com/) and load in some fake text. 
# You'll need to add some kind of text to load to make this work. We'll use LangChain's TextLoader to load this data.

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# We have some fake laptop reviews that we can load in. Use some kind of alt text to make this work
laptop_reviews = TextLoader(
    f"{DA.paths.datasets}/reviews/fake_laptop_reviews.txt", encoding="utf8"
)
document = laptop_reviews.load() # We also load fake customer reviews
display(document)

### Step 2 - Chunking and Embeddings
# After the data is in document format, split data into chunks using a CharacterTextSplitter 
# Then embed this data using Hugging Face's embedding LLM to embed this data for our vector store.

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# First we split the data into manageable chunks to store as vectors. There isn't an exact way to do this, more chunks means more detailed context, but will increase the size of our vectorstore.
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
texts = text_splitter.split_documents(document)
# Now we'll create embeddings for our document so we can store it in a vector store and feed the data into an LLM. We'll use the sentence-transformers model for out embeddings. https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, cache_folder=DA.paths.datasets
)  # Use a pre-cached model
# Finally we make our Index using chromadb and the embeddings LLM
chromadb_index = Chroma.from_documents(
    texts, embeddings, persist_directory=DA.paths.working_dir
)

### Step 3 - Creating our Document QA LLM Chain
# With our data now in vector form we need an LLM and a chain to take our queries and create tasks for our LLM to perform.

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# We want to make this a retriever, so we need to convert our index.  This will create a wrapper around the functionality of our vector database so we can search for similar documents/chunks in the vectorstore and retrieve the results:
retriever = chromadb_index.as_retriever()

# This chain will be used to do QA on the document. We will need
# 1 - A LLM to do the language interpretation
# 2 - A vector database that can perform document retrieval
# 3 - Specification on how to deal with this data (more on this soon)

hf_llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={
        "temperature": 0,
        "max_length": 128,
        "cache_dir": DA.paths.datasets,
    },
)

chain_type = "stuff"  # Options: stuff, map_reduce, refine, map_rerank
laptop_qa = RetrievalQA.from_chain_type(
    llm=hf_llm, chain_type="stuff", retriever=retriever
)

### Step 4 - Talking to Our Data
# Now we are ready to send prompts to our LLM and have it use our prompt, the access to our data, and read the information, process, and return with a response.

# Let's ask the chain about the product we have.
laptop_name = laptop_qa.run("What is the full name of the laptop?")
display(laptop_name)
# example response: 'Raytech Supernova'

# Now we'll ask the chain about the product.
laptop_features = laptop_qa.run("What are some of the laptop's features?")
display(laptop_features)
# example response: 'The 4K display, powerful GPU, and fast SSD'

# Finally let's ask the chain about the reviews.
laptop_reviews = laptop_qa.run("What is the general sentiment of the reviews?")
display(laptop_reviews)
# example response: 'positive'

###END OF EXAMPLE 
