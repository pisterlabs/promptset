#%%
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import StreamingStdOutCallbackHandler

import time
import os

from dotenv import load_dotenv
load_dotenv('../.env')

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ]
)
#%%
start_tick = time.time()
cache_dir = LocalFileStore("./.cache/")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
loader = UnstructuredFileLoader("../res/1984_chapter_1.txt")

docs = loader.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vectorstore = FAISS.from_documents(docs, cached_embeddings)

retriver = vectorstore.as_retriever()
print(f"Vectorstore loaded in {time.time() - start_tick:.2f} seconds")

#%%

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:
            
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

result = chain.invoke("Describe Victory Mansions")

print(type(result))
# %%
