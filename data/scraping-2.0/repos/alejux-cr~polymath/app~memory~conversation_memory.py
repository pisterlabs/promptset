# from langchain.memory import ConversationBufferMemory
# from langchain.memory import ConversationTokenBufferMemory
# from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.memory import ConversationSummaryBufferMemory
# from langchain.memory import ConversationEntityMemory
# from langchain.memory import ConversationKGMemory
# from langchain.memory import VectorStoreRetrieverMemory

from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

# Simplest form of memory only saved in buffer
# This memory allows for storing of messages and then extracts the messages in a variable.
# memory = ConversationBufferMemory()

#  keeps a list of the interactions of the conversation over time. It only uses the last K interactions.
# This can be useful for keeping a sliding window of the most recent interactions, so the buffer does not get too large
# memory = ConversationBufferWindowMemory(k=10)

# Entity Memory remembers given facts about specific entities in a conversation.
# It extracts information on entities (using an LLM) and builds up its knowledge about that entity over time (also using an LLM).
# memory = ConversationEntityMemory(llm=llm)

# This type of memory uses a knowledge graph to recreate memory.
# memory = ConversationKGMemory(llm=llm)

# This type of memory creates a summary of the conversation over time. This can be useful for condensing information 
# from the conversation over time. Conversation summary memory summarizes the conversation as it happens and 
# stores the current summary in memory. This memory can then be used to inject the summary of the conversation so far 
# into a prompt/chain. This memory is most useful for longer conversations, where keeping the past message history 
# in the prompt verbatim would take up too many tokens.
# memory = ConversationSummaryMemory(llm=chat)

# It keeps a buffer of recent interactions in memory, but rather than just completely flushing old interactions it 
# compiles them into a summary and uses both. Unlike the previous implementation though, it uses token length 
# rather than number of interactions to determine when to flush interactions.
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)

# keeps a buffer of recent interactions in memory, and uses token length rather than number of interactions
# to determine when to flush interactions.
# memory = ConversationTokenBufferMemory(llm=chat, max_token_limit=60)


# Vector store-backed memory
# stores memories in a VectorDB and queries the top-K most "salient" docs every time it is called.
# This differs from most of the other Memory classes in that it doesn't explicitly track the order of interactions.
# In this case, the "docs" are previous conversation snippets. This can be useful to refer to relevant pieces of
# information that the AI was told earlier in the conversation.

embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
retriever = vectorstore.as_retriever(search_kwargs=dict(k=10))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# When added to an agent, the memory object can save pertinent information from conversations or used tools
memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})
memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"})
print(memory.load_memory_variables({"prompt": "what sport should i watch?"})["history"])


