# Check out the docs:
# - [LangChain](https://docs.langchain.com/docs/)
# - [OpenAI](https://github.com/openai/openai-python)
# - [Pinecone](https://docs.pinecone.io/docs/overview)


# %%
import os
from dotenv import load_dotenv

load_dotenv()

# %%
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# %%
review_df = pd.read_csv("./data/data_2.csv")

# %%
data = review_df

# %% [markdown]
# #### Data Parsing
# 
# Now that we have our data - let's go ahead and set up some tools to parse it into a more usable format for LangChain!

# %% [markdown]
# Our reviews might contain a lot of information, and in order to ensure they don't exceed the context window of our model and to allow us to include a few reviews as context for each query - let's construct a system to "chunk" our data into smaller pieces.
# 
# We'll be leveraging the `RecursiveCharacterTextSplitter` for this task today.
# 
# While splitting our text seems like a simple enough task - getting this correct/incorrect can have massive downstream impacts on your application's performance.
# 
# You can read the docs here:
# - [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter)
# 
# > ### HINT:
# >It's always worth it to check out the LangChain source code if you're ever in a bind - for instance, if you want to know how to transform a set of documents, check it out [here](https://github.com/langchain-ai/langchain/blob/5e9687a196410e9f41ebcd11eb3f2ca13925545b/libs/langchain/langchain/text_splitter.py#L268C18-L268C18)

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, # the character length of the chunk
    chunk_overlap = 100, # the character length of the overlap between chunks
    length_function = len, # the length function - in this case, character length (aka the python len() fn.)
)

# %% [markdown]
# Now that we have our `RecursiveCharacterTextSplitter` set up - let's look at how it might split our source text. 
# 
# Keep in mind that the source text is split by `["\n\n", "\n", " ", ""]` in that order.
# 
# We know that each of the subheadings in our review `page_content` is separated by a newline character, so it will preferably chunk the review subheadings together. 
# 
# That's great! Let's move on to creating our index!

# %% [markdown]
# ### Task 2: Creating an "Index"
# 
# The term "index" is used largely to mean: Structured documents parsed into a useful format for querying, retrieving, and use in the LLM application stack.

# %% [markdown]
# #### Selecting Our VectorStore
# 
# There are a number of different VectorStores, and a number of different strengths and weaknesses to each.
# 
# In this notebook, we will be keeping it very simple by leveraging Pinecone's API Vector Database.

# %% [markdown]
# Let's set up a Pinecone index using the methods provided in their [documentation](https://docs.pinecone.io/docs/langchain)!

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.dataframe import DataFrameLoader

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 500,
#     chunk_overlap = 100
# )

loader = DataFrameLoader(review_df, page_content_column="facts")
base_docs = loader.load()
docs = text_splitter.split_documents(base_docs)


embedder = OpenAIEmbeddings()

# Embed and persist db
persist_directory = "./data/chroma"

# Only use this when you create the db to store/persist data 
# vectorstore = Chroma.from_documents(documents=docs, persist_directory=persist_directory, embedding=embedder)

# vectorstore = vectordb.from_documents(
#     documents=docs,
#     embedding=embedder)
# vectorstore.persist()

vectorstore = None
# Read from disc
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedder)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

primary_qa_llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k", 
    temperature=0,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=primary_qa_llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

# query = "Give me info on "

# result = qa_chain({"query" : query})

# print(result["result"])


# # %%
# import pinecone
# # from pinecone import GRPCIndex

# YOUR_API_KEY = os.environ["PINECONE_API_KEY"]
# YOUR_ENV = os.environ["PINECONE_ENV"]

# index_name = 'barbie-review-index'

# pinecone.init(
#     api_key=YOUR_API_KEY,
#     environment=YOUR_ENV
# )

# if index_name not in pinecone.list_indexes():
#     # we create a new index
#     pinecone.create_index(
#         name=index_name,
#         metric='cosine',
#         dimension=1536
#     )

# # %% [markdown]
# # Now we can connect to our index and view some statistics about it.

# # %%
# # index = pinecone.GRPCIndex(index_name)
# index = pinecone.Index(index_name)

# index.describe_index_stats()

# # %% [markdown]
# # We're going to be setting up our VectorStore with the OpenAI embeddings model. While this embeddings model does not need to be consistent with the LLM selection, it does need to be consistent between embedding our index and embedding our queries over that index.
# # 
# # While we don't have to worry too much about that in this example - it's something to keep in mind for more complex applications.
# # 
# # We're going to leverage a [`CacheBackedEmbeddings`](https://python.langchain.com/docs/modules/data_connection/caching_embeddings )flow to prevent us from re-embedding similar queries over and over again.
# # 
# # Not only will this save time, it will also save us precious embedding tokens, which will reduce the overall cost for our application.
# # 
# # >#### Note:
# # >The overall cost savings needs to be compared against the additional cost of storing the cached embeddings for a true cost/benefit analysis. If your users are submitting the same queries often, though, this pattern can be a massive reduction in cost.

# # %%
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import CacheBackedEmbeddings
# from langchain.storage import LocalFileStore

# store = LocalFileStore("./cache/")

# core_embeddings_model = OpenAIEmbeddings()

# embedder = CacheBackedEmbeddings.from_bytes_store(
#     core_embeddings_model, store, namespace=core_embeddings_model.model
# )

# %% [markdown]
# Now that we have our `CacheBackedEmbeddings` pipeline set-up, let's index our documents into our Pinecone Vector Database. 
# 
# We'll add some useful metadata as well!

# # %%
# data.head(1)

# # %%
# from tqdm.auto import tqdm
# from uuid import uuid4

# BATCH_LIMIT = 100

# texts = []
# metadatas = []

# for i in tqdm(range(len(data))):

#     record = data.iloc[i]

#     metadata = {
#         'review-url': str(record["Review_Url"]),
#         'review-date' : str(record["Review_Date"]),
#         'author' : str(record["Author"]),
#         'rating' : str(record["Rating"]),
#         'review-title' : str(record["Review_Title"]),
#     }

#     record_texts = text_splitter.split_text(record["Review"])

#     record_metadatas = [{
#         "chunk": j, "text": text, **metadata
#     } for j, text in enumerate(record_texts)]
#     texts.extend(record_texts)
#     metadatas.extend(record_metadatas)
#     if len(texts) >= BATCH_LIMIT:
#         ids = [str(uuid4()) for _ in range(len(texts))]
#         embeds = embedder.embed_documents(texts)
#         index.upsert(vectors=zip(ids, embeds, metadatas))
#         texts = []
#         metadatas = []

# if len(texts) > 0:
#     ids = [str(uuid4()) for _ in range(len(texts))]
#     embeds = embedder.embed_documents(texts)
#     index.upsert(vectors=zip(ids, embeds, metadatas))

# # %%
# index.describe_index_stats()

# # %% [markdown]
# # Now that we've created our index, let's convert it to a LangChain `VectorStroe` so we can use it in the rest of the LangChain ecosystem!

# # %%
# from langchain.vectorstores import Pinecone

# text_field = "text"

# index = pinecone.Index(index_name)

# vectorstore = Pinecone(
#     index, embedder.embed_query, text_field
# )

# %% [markdown]
# Now that we've created the VectorStore, we can check that it's working by embedding a query and retrieving passages from our reviews that are close to it.

# %%
query = "child support case"

d = vectorstore.similarity_search(
    query, 
    k=3  
)
print(d)

# %% [markdown]
# Let's see how much time the `CacheBackedEmbeddings` pattern saves us:

# %%
# %%timeit
# query = "I really wanted to enjoy this and I know that I am not the target audience but there were massive plot holes and no real flow."
# vectorstore.similarity_search(
#     query, 
#     k=3  
# )

# %% [markdown]
# As we can see, even over a significant number of runs - the cached query is significantly faster than the first instance of the query!
# 
# With that, we're ready to move onto Task 3!

# %% [markdown]
# ### Task 3: Building a Retrieval Chain
# 
# In this task, we'll be making a Retrieval Chain which will allow us to ask semantic questions over our data.
# 
# This part is rather abstracted away from us in LangChain and so it seems very powerful.
# 
# Be sure to check the documentation, the source code, and other provided resources to build a deeper understanding of what's happening "under the hood"!

# %% [markdown]
# #### A Basic RetrievalQA Chain
# 
# We're going to leverage `return_source_documents=True` to ensure we have proper sources for our reviews - should the end user want to verify the reviews themselves.
# 
# Hallucinations [are](https://arxiv.org/abs/2202.03629) [a](https://arxiv.org/abs/2305.15852) [massive](https://arxiv.org/abs/2303.16104) [problem](https://arxiv.org/abs/2305.18248) in LLM applications.
# 
# Though it has been tenuously shown that using Retrieval Augmentation [reduces hallucination in conversations](https://arxiv.org/pdf/2104.07567.pdf), one sure fire way to ensure your model is not hallucinating in a non-transparent way is to provide sources with your responses. This way the end-user can verify the output.

# %% [markdown]
# #### Our LLM
# 
# In this notebook, we'll continue to leverage OpenAI's suite of models - this time we'll be using the `gpt-3.5-turbo` model to power our RetrievalQAWithSources chain.

# %%
from langchain.llms.openai import OpenAIChat

llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0)

# %% [markdown]
# Now we can set up our chain.

# %%
retriever = vectorstore.as_retriever()

# %%
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage
from chainlit.playground.providers.openai import ChatOpenAI  # importing ChatOpenAI tools


handler = StdOutCallbackHandler()

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():

    qa_with_sources_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        callbacks=[handler],
        return_source_documents=True
    )

    cl.user_session.set("qa_with_sources_chain", qa_with_sources_chain)


# # %%
# qa_with_sources_chain({"query" : "How was Will Ferrell in this movie?"})["result"]

# # %%
# qa_with_sources_chain({"query" : "Do reviewers consider this movie Kenough?"})["result"]

# # %% [markdown]
# # Let's look at the available metadata we have, thanks to our index-creation!

# # %%
# result = qa_with_sources_chain({"query" : "Was Will Ferrel funny?"})

# # %%
# for k, v in result.items():
#     print(f"Key: {k}")
#     print(f"Value: {v}")
#     print("")

# # %%
# for page_content, metadata in result["source_documents"]:
#     print(f"Metadata: {metadata}")
#     print(f"Page Content: {page_content}")
#     print("")

# %% [markdown]
# ### Adding Prompt Caching and Monitoring
# 
# Now that we have the basic `RetrievalQAChain` set up and working - let's add a few more tools to help us built a more performant application and add a visibility tool as well!

# %% [markdown]
# #### Visibility Tooling
# 
# We'll be once again leveraging Weights and Biases as our visibility tool, so let's add that first!
# 
# You'll want to use the same Weights and Biases account that you set-up last Thursday here!

# %%
# os.environ["WANDB_API_KEY"] = getpass.getpass("Weights and Biases API Key:")
# os.environ["WANDB_PROJECT"] = "barbie-retrieval-qa"

# %% [markdown]
# Now, to set up WandB, all we have to do is...

# %%
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

# %% [markdown]
# Yes, that's it. 
# 
# Let's use our `RetrievalQA` chain to test it out!

# %%
# qa_with_sources_chain({"query" : "Do reviewers consider this movie Kenough?"})["result"]

# %% [markdown]
# With those simple lines of code - we've added full visibility to our prompts and responses through Weights and Biases! 

# %% [markdown]
# #### Prompt Caching
# 

# %% [markdown]
# ### Adding A Prompt Cache
# 
# The basic idea of Prompt Caching is to provide a way to circumvent going to the LLM for prompts we have already seen.
# 
# Similar to cached embeddings, the idea is simple:
# 
# - Keep track of all the input/output pairs
# - If a user query is (in the case of semantic similarity caches) close enough to a previous prompt contained in the cache, return the output associated with that pair

# %% [markdown]
# ### Initializing a Prompt Cache
# 
# There are many different tools you can use to implement a Prompt Cache - from a "build it yourself" VectorStore implementation - to Redis - to custom libraries - there are upsides and downsides to each solution. 
# 
# Let's look at the Redis-backed Cache vs. `InMemoryCache` as an example:
# 
# Redis Cache
# | Pros  | Cons  |
# |---|---|
# | Managed and Robust  | Expensive to Host  |
# | Integrations on all Major Cloud Platforms  | Non-trivial to Integrate |
# | Easily Scalable  | Does not have a ChatModel implementation |
# 
# `InMemoryCache`
# | Pros  | Cons  |
# |---|---|
# | Easily implemented  | Consumes potentially precious memory |
# | Completely Cloud Agnostic  | Does not offer inter-session caching |
# 
# For the sake of ease of use - and to allow functionality with our `ChatOpenAI` model - we'll leverage `InMemoryCache`.

# %% [markdown]
# We need to set our `langchain.llm_cache` to use the `InMemoryCache`.
# 
# - [`InMemoryCache`](https://api.python.langchain.com/en/latest/cache/langchain.cache.InMemoryCache.html)

# %%
# import langchain
# from langchain.cache import InMemoryCache
# langchain.llm_cache = InMemoryCache()

# %% [markdown]
# One more important fact about the `InMemoryCache` is that it is what's called an "exact-match" cache - meaning it will only trigger when the user query is *exactly* represented in the cache. 
# 
# This is a safer cache, as we can guarentee the user's query exactly matches with previous queries and we don't have to worry about edge-cases where semantic similarity might fail - but it does reduce the potential to hit the cache.
# 
# We could leverage tools like `GPTCache`, or `RedisCache` (for non-chat model implementations) to get a "semantic similarity" cache, if desired!

# %%
# %%time
# qa_with_sources_chain({"query" : "Do reviewers consider this movie Kenough?"})["result"]

# %%
# %%time
# qa_with_sources_chain({"query" : "Do reviewers consider this movie Kenough?"})["result"]

# %% [markdown]
# Let's look at an example that is extremely close - but is not the exact query.

# %%
# %%time
# qa_with_sources_chain({"query" : "Do reviewers consider this here movie Kenough?"})["result"]

# %% [markdown]
# As you can see, adding an exact-match prompt cache is a very small lift - but it can significantly improve the latency of your end-user application experience!

# %% [markdown]
# ### Conclusion
# 
# And with that, we have our Barbie Review RAQA Application built!
# 
# Let's port it into a Chainlit app and put it up on a Hugging Face Space!




#####################################
# Chainlit for Huggingface setup
#####################################

@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: str):
    qa_with_sources_chain = cl.user_session.get("qa_with_sources_chain")

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                # template=RAQA_PROMPT_TEMPLATE,
                # formatted=RAQA_PROMPT_TEMPLATE,
            ),
            PromptMessage(
                role="user",
                # template=user_template,
                # formatted=user_template.format(input=message),
            ),
        ],
        inputs={"input": message},
        # settings=settings,
    )

    msg = cl.Message(content="")

    result = await qa_with_sources_chain.acall({"query" : message})  #, callbacks=[cl.AsyncLangchainCallbackHandler()])

    print(result)


    # Update the prompt object with the completion
    msg.content = result["result"]
    prompt.completion = msg.content
    msg.prompt = prompt
    await msg.send()

    pass
