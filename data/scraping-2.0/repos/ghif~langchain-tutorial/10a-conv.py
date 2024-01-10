from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import json
import copy
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import (
    PromptTemplate,
    SemanticSimilarityExampleSelector
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


template = """You are a knowledgeable customer service agent from Pusat Bantuan Merdeka Belajar Kampus Merdeka (MBKM).
Use the historical conversation below to answer various questions from users.
If you don't know the answer, just say I don't know. Don't make up an answer.
The answer given must always be in Indonesian with a friendly tone.

Current conversation:
{chat_history}

Human: {input}
AI Assistant:"""


# LLM
chat_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    verbose=True,
    temperature=0.0,
    streaming=True
)

# Create prompt template
prompt = PromptTemplate.from_template(template)

# Define memory
memory = ConversationBufferMemory(
    llm=chat_llm,
    memory_key="chat_history",
    human_prefix="Human",
    ai_prefix="AI Assistant",
    return_messages=True
)

# Chain
chain = ConversationChain(
    prompt=prompt,
    llm=chat_llm, 
    memory=memory,
    verbose=True
)

query = "Halo, kamu dengan siapa?"
print(query)
response = chain.predict(input=query)
print(response)

query = "Tolong jelaskan mengenai program MSIB (Magang dan Studi Independen Bersertifikat)."
print(query)
response = chain.predict(input=query)
print(response)

query = "Apa nama program yang saya tanyakan sebelumnya?"
print(query)
response = chain.predict(input=query)
print(response)

