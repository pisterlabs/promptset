#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKey')


# In[2]:


from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader


# In[3]:


# Loading a single website
loader = WebBaseLoader("http://www.paulgraham.com/wealth.html")
docs = loader.load()

# Split your website into big chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

print (f"Your {len(docs)} documents have been split into {len(chunks)} chunks")


# In[4]:


embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding)


# In[5]:


retriever_vanilla = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 8})

retriever_mmr = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8})


# In[6]:


vanilla_relevant_docs = retriever_vanilla.get_relevant_documents("What is the best way to make and keep wealth?")


# In[7]:


mmr_relevant_docs = retriever_mmr.get_relevant_documents("What is the best way to make and keep wealth?")


# In[8]:


def analyze_list_overlap(list1, list2, content_attr='page_content'):
    """
    Analyze the overlap and uniqueness between two lists of objects using a specified content attribute.

    Parameters:
    list1 (list): The first list of objects to compare.
    list2 (list): The second list of objects to compare.
    content_attr (str): The attribute name of the content to use for comparison.

    Returns:
    dict: A dictionary with counts of overlapping, unique to list1, unique to list2 items,
          and total counts for each list.
    """
    # Extract unique content attributes from the lists
    set1_contents = {getattr(doc, content_attr) for doc in list1}
    set2_contents = {getattr(doc, content_attr) for doc in list2}

    # Find the number of overlapping content attributes
    overlap_contents = set1_contents & set2_contents
    overlap_count = len(overlap_contents)

    # Find the unique content attributes in each list
    unique_to_list1_contents = set1_contents - set2_contents
    unique_to_list2_contents = set2_contents - set1_contents
    unique_to_list1_count = len(unique_to_list1_contents)
    unique_to_list2_count = len(unique_to_list2_contents)

    # Use the unique content attributes to retrieve the unique objects
    unique_to_list1 = [doc for doc in list1 if getattr(doc, content_attr) in unique_to_list1_contents]
    unique_to_list2 = [doc for doc in list2 if getattr(doc, content_attr) in unique_to_list2_contents]

    # Count the total number of items in each list
    total_list1 = len(list1)
    total_list2 = len(list2)

    # Return the results in a dictionary
    return {
        'total_list1': total_list1,
        'total_list2': total_list2,
        'overlap_count': overlap_count,
        'unique_to_list1_count': unique_to_list1_count,
        'unique_to_list2_count': unique_to_list2_count,
    }


# In[9]:


analyze_list_overlap(vanilla_relevant_docs, mmr_relevant_docs)


# In[ ]:




