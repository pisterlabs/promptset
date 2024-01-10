#!/usr/bin/env python
# coding: utf-8

# # Advanced Retrieval With LangChain
# 
# Let's go over a few more complex and advanced retrieval methods with LangChain.
# 
# There is no one right way to retrieve data - it'll depend on your application so take some time to think about it before you jump in
# 
# Let's have some fun
# 
# * **Multi Query** - Given a single user query, use an LLM to synthetically generate multiple other queries. Use each one of the new queries to retrieve documents, take the union of those documents for the final context of your prompt
# * **Contextual Compression** - Fluff remover. Normal retrieval but with an extra step of pulling out relevant information from each returned document. This makes each relevant document smaller for your final prompt (which increases information density)
# * **Parent Document Retriever** - Split and embed *small* chunks (for maximum information density), then return the parent documents (or larger chunks) those small chunks come from
# * **Ensemble Retriever** - Combine multiple retrievers together
# * **Self-Query** - When the retriever infers filters from a users query and applies those filters to the underlying data

# In[1]:


from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKey')


# ## Load up our texts and documents
# 
# Then chunk them, and put them into a vector store

# In[2]:


from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


# We're going to load up Paul Graham's essays. In this repo there are various sizes of folders (`PaulGrahamEssaysSmall`, `PaulGrahamEssaysMedium`, `PaulGrahamEssaysLarge` or `PaulGrahamEssays` for the full set.)

# In[3]:


loader = DirectoryLoader('../data/PaulGrahamEssaysLarge/', glob="**/*.txt", show_progress=True)

docs = loader.load()


# In[4]:


print (f"You have {len(docs)} essays loaded")


# Then we'll split up our text into smaller sized chunks

# In[5]:


# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
splits = text_splitter.split_documents(docs)

print (f"Your {len(docs)} documents have been split into {len(splits)} chunks")

If you do `Chroma.from_documents` multiple times you'll re-add the documents (and duplicate them) which is annoying. I check to see if we've already made our vectordb, if so delete what's in there, then go and make it
# In[6]:


if 'vectordb' in globals(): # If you've already made your vectordb this will delete it so you start fresh
    vectordb.delete_collection()

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)


# ### MultiQuery
# 
# This retrieval method will generated 3 additional questions to get a total of 4 queries (with the users included) that will be used to go retrieve documents. This is helpful when you want to retrieve documents which are similar in meaning to your question.

# In[7]:


from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
# Set logging for the queries
import logging


# Doing some logging to see the other questions that were generated. I tried to find a way to get these via a model property but couldn't, lmk if you find a way!

# In[8]:


logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# Then we set up the MultiQueryRetriever which will generate other questions for us

# In[9]:


question = "What is the authors view on the early stages of a startup?"
llm = ChatOpenAI(temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)


# In[10]:


unique_docs = retriever_from_llm.get_relevant_documents(query=question)


# Check out how there are other questions which are related to but slightly different than the question I asked.
# 
# Let's see how many docs were actually returned

# In[11]:


len(unique_docs)


# Ok now let's put those docs into a prompt template which we'll use as context

# In[12]:


prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# In[13]:


llm.predict(text=PROMPT.format_prompt(
    context=unique_docs,
    question=question
).text)


# ### Contextual Compression
# 
# Then we'll move onto contextual compression. This will take the chunk that you've made (above) and compress it's information down to the parts relevant to your query.
# 
# Say that you have a chunk that has 3 topics within it, you only really care about one of them though, this compressor will look at your query, see that you only need one of the 3 topics, then extract & return that one topic.
# 
# This one is a bit more expensive because each doc returned will get processed an additional time (to pull out the relevant data)

# In[14]:


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# We first need to set up our compressor, it's cool that it's a separate object because that means you can use it elsewhere outside this retriever as well.

# In[15]:


llm = ChatOpenAI(temperature=0, model='gpt-4')

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                       base_retriever=vectordb.as_retriever())


# First, an example of compression. Below we have one of our splits that we made above

# In[16]:


splits[0].page_content


# Now we are going to pass a question to it and with that question we will compress the doc. The cool part is this doc will be contextually compressed, meaning the resulting file will only have the information relevant to the question.

# In[17]:


compressor.compress_documents(documents=[splits[0]], query="test for what you like to do")


# Great so we had a long document, now we have a shorter document with more dense information. Great for getting rid of the fluff. Let's try it out on our essays

# In[18]:


question = "What is the authors view on the early stages of a startup?"
compressed_docs = compression_retriever.get_relevant_documents(question)


# In[19]:


print (len(compressed_docs))
compressed_docs


# We now have 4 docs but they are shorter and only contain the information that is relevant to our query.
# 
# Let's put it in our prompt template again.

# In[20]:


prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# In[21]:


llm.predict(text=PROMPT.format_prompt(
    context=compressed_docs,
    question=question
).text)


# ### Parent Document Retriever
# 
# [LangChain documentation](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever) does a great job describing this - my minor edits below:
# 
# When you split your docs, you generally may want to have small documents, so that their embeddings can most accurately reflect their meaning. If too long, then the embeddings can lose meaning.
# 
# But at the same time you may want to have information around those small chunks to keep context of the longer document.
# 
# The ParentDocumentRetriever strikes that balance by splitting and storing small chunks of data. During retrieval, it first fetches the small chunks but then looks up the parent ids for those chunks and returns those larger documents.
# 
# Note that "parent document" refers to the document that a small chunk originated from. This can either be the whole raw document OR a larger chunk.

# In[22]:


from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore


# In[23]:


# This text splitter is used to create the child documents. They should be small chunk size.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)


# In[24]:


# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="return_full_documents",
    embedding_function=OpenAIEmbeddings()
)


# In[25]:


# The storage layer for the parent documents
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, 
    docstore=store, 
    child_splitter=child_splitter,
)


# Now we will add the whole essays that we split above. We haven't chunked these essays yet, but the `.add_documents` will do the small chunking for us with the `child_splitter` above

# In[26]:


retriever.add_documents(docs, ids=None)


# Now if we were to put in a question or query, we'll get small chunks returned

# In[27]:


sub_docs = vectorstore.similarity_search("what is some investing advice?")


# In[28]:


sub_docs


# Look how small those chunks are. Now we want to get the parent doc which those small docs are a part of.

# In[29]:


retrieved_docs = retriever.get_relevant_documents("what is some investing advice?")


# I'm going to only do the first doc to save space, but there are more waiting for you. Keep in mind that LangChain will do the union of docs, so if you have two child docs from the same parent doc, you'll only return the parent doc once, not twice.

# In[30]:


retrieved_docs[0].page_content[:1000]


# However here we got the full document back. Sometimes this will be too long and we actually just want to get a larger chunk instead. Let's do that.
# 
# Notice the chunk size difference between the parent splitter and child splitter.

# In[31]:


# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="return_split_parent_documents", embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()


# This will set up our retriever for us

# In[32]:


retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, 
    docstore=store, 
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)


# Now this time when we add documents two things will happen
# 1. Larger chunks - We'll split our docs into large chunks
# 2. Smaller chunks - We'll split our docs into smaller chunks
# 
# Both of them will be combined.

# In[33]:


retriever.add_documents(docs)


# Let's check out how many documents we have now

# In[34]:


len(list(store.yield_keys()))


# Then let's go get our small chunks to make sure it's working and see how long they are

# In[35]:


sub_docs = vectorstore.similarity_search("what is some investing advice?")
sub_docs


# Now, let's do the full process, we'll see what small chunks are generated, but then return the larger chunks as our relevant documents

# In[36]:


larger_chunk_relevant_docs = retriever.get_relevant_documents("what is some investing advice?")
larger_chunk_relevant_docs[0]


# In[37]:


prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

question = "what is some investing advice?"

llm.predict(text=PROMPT.format_prompt(
    context=larger_chunk_relevant_docs,
    question=question
).text)


# ### Ensemble Retriever
# 
# The next one on our list combines multiple retrievers together. The goal here is to see what multiple methods return, then pull them together for (hopefully) better results.
# 
# You may need to install bm25 with `!pip install rank_bm25`

# In[38]:


from langchain.retrievers import BM25Retriever, EnsembleRetriever


# We'll use a [BM25 retriever](https://en.wikipedia.org/wiki/Okapi_BM25) for this one which is really good at keyword matching (vs semantic). When you combine this method with regular semantic search it's known as hybrid search.

# In[39]:


# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 2


# In[40]:


embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(splits, embedding)
vectordb = vectordb.as_retriever(search_kwargs={"k": 2})


# In[41]:


# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vectordb], weights=[0.5, 0.5])


# In[42]:


ensemble_docs = ensemble_retriever.get_relevant_documents("what is some investing advice?")
len(ensemble_docs)


# In[43]:


prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

question = "what is some investing advice?"

llm.predict(text=PROMPT.format_prompt(
    context=ensemble_docs,
    question=question
).text)


# ### Self Querying
# 
# The last one we'll look at today is self querying. This is when the retriever has the ability to query itself. It does this so it can use filters when doing it's final query.
# 
# This means it'll use the users query for semantic search, but also its own query for filtering (so the user doesn't have to give a structured filter).
# 
# You may need to install `!pip install lark`

# In[44]:


from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model='gpt-4')


# In[45]:


if 'vectorstore' in globals(): # If you've already made your vectordb this will delete it so you start fresh
    vectorstore.delete_collection()

vectorstore = Chroma.from_documents(
    splits, embeddings
)


# Below is the information on the fitlers available. This will help the model know which filters to semantically search for

# In[46]:


metadata_field_info=[
    AttributeInfo(
        name="source",
        description="The filename of the essay", 
        type="string or list[string]", 
    ),
]


# In[47]:


document_content_description = "Essays from Paul Graham"
retriever = SelfQueryRetriever.from_llm(llm,
                                        vectorstore,
                                        document_content_description,
                                        metadata_field_info,
                                        verbose=True,
                                        enable_limit=True)


# In[48]:


retriever.get_relevant_documents("Return only 1 essay. What is one thing you can do to figure out what you like to do from source '../data/PaulGrahamEssaysLarge/island.txt'")


# It's kind of annoying to have to put in the full file name, a user doesn't want to do that. Let's change `source` to `essay` and the file path w/ the essay name

# In[49]:


import re

for split in splits:
    split.metadata['essay'] = re.search(r'[^/]+(?=\.\w+$)', split.metadata['source']).group()


# Ok now that we did that, let's make a new field info config

# In[50]:


metadata_field_info=[
    AttributeInfo(
        name="essay",
        description="The name of the essay", 
        type="string or list[string]", 
    ),
]


# In[51]:


if 'vectorstore' in globals(): # If you've already made your vectordb this will delete it so you start fresh
    vectorstore.delete_collection()

vectorstore = Chroma.from_documents(
    splits, embeddings
)


# In[52]:


document_content_description = "Essays from Paul Graham"
retriever = SelfQueryRetriever.from_llm(llm,
                                        vectorstore,
                                        document_content_description,
                                        metadata_field_info,
                                        verbose=True,
                                        enable_limit=True)


# In[53]:


retriever.get_relevant_documents("Tell me about investment advice the 'worked' essay? return only 1")


# Awesome! It returned it back for us. It's a bit rigid because you need to put in the exact name of the file/essay you want to get. You could make a pre-step and infer the correct essay from the users choice but this is out of scope for now and application specific.
