from langchain.chains import base

from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain,RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import faiss
import openai

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# # When added to an agent, the memory object can save pertinent information from conversations or used tools
memory.save_context({"input": "My problem is I very sad due to a breakup with my girlfriend"}, {"output": "Oh, I'm so sorry to hear that, that can be really tough. Want to talk about it? Im happy to lend an ear. Was it a mutual decision or a difficult break up?"})
# memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
# memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"}) #