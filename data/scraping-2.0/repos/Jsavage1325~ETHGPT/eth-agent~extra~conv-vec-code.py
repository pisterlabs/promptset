from datetime import datetime

import faiss
from langchain.chains import ConversationChain
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
air_vectorstore = FAISS.load_local("airstack_baby_faiss_index", embeddings)


embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

# In actual usage, you would set `k` to be a higher value, but we use k=1 to show that
# the vector lookup still returns the semantically relevant information
# retriever = vectorstore.as_retriever(search_kwargs=dict(k=10))
retriver = air_vectorstore.as_retriever(search_kwargs=dict(k=5))
memory = VectorStoreRetrieverMemory(retriever=retriver)

# When added to an agent, the memory object can save pertinent information from conversations or used tools
memory.save_context(
    {"input": "how do you write execute eth commands"},
    {"output": "by writing web3.py code"},
)


llm = OpenAI(temperature=0)  # Can be any valid LLM
_DEFAULT_TEMPLATE = """
you are a natural language interface to ethereum you write python code to execute commands on the ethereum blockchain. write the relivent code to do what the human wants

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    # We set a very low max_token_limit for the purposes of testing.
    memory=memory,
    verbose=True,
)
print(conversation_with_summary.predict(input="can you send all my eth to spink.eth"))
