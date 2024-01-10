#!/usr/bin/env python
# coding: utf-8

# In[14]:


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk

openai_api_key = os.getenv("OPENAI_API_KEY", "YourAPIKey")

# nltk.download('averaged_perceptron_tagger')

# pip install unstructured
# Other dependencies to install https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
# pip install python-magic-bin
# pip install chromadb


# In[2]:


# Get your loader ready
loader = DirectoryLoader('../data/PaulGrahamEssaySmall/', glob='**/*.txt')


# In[3]:


# Load up your text into documents
documents = loader.load()


# In[4]:


# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


# In[5]:


# Split your documents into texts
texts = text_splitter.split_documents(documents)


# In[6]:


# Turn your texts into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


# In[7]:


# Get your docsearch ready
docsearch = FAISS.from_documents(texts, embeddings)


# In[8]:


# Load up your LLM
llm = OpenAI(openai_api_key=openai_api_key)


# In[9]:


# Create your Retriever
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())


# In[10]:


# Run a query
query = "What did McCarthy discover?"
qa.run(query)


# ### Sources

# In[11]:


qa = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=docsearch.as_retriever(),
                                return_source_documents=True)
query = "What did McCarthy discover?"
result = qa({"query": query})


# In[12]:


result['result']


# In[13]:


result['source_documents']

