from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, SemanticSimilarityExampleSelector
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

import json

# Context prompt
# template = """You are a knowledgeable customer service agent from Pusat Bantuan Merdeka Belajar Kampus Merdeka (MBKM).
# If you don't know the answer, just say I don't know. Don't make up an answer.
# The answer given must always be in Indonesian with a friendly tone.

# Human: {input}
# AI Assistant:"""

prefix = """You are a knowledgeable customer service from Pusat Bantuan Merdeka Belajar Kampus Merdeka (MBKM).
Use the context below to answer various questions from users.
If you don't know the answer, just say I don't know. Don't make up an answer.
The answer given must always be in Indonesian language with a friendly tone.

Here are some examples of conversations between users and customer service to be your references:
"""

example_template = """
Question: {question}
Answer: {answer}
"""

suffix = """
Question: {input}
Answer: 
"""

# LLM
chat_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# # Prompt 
# prompt = PromptTemplate.from_template(template)


# Activate Retrieval Augmented Generation (RAG)

# load few shot conversation examples
examples = json.load(open("chat_samples_withbubble.json", "r"))

example_prompt = PromptTemplate.from_template(example_template)

embeddings = OpenAIEmbeddings()
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, 
    embeddings,
    FAISS, 
    k=3 # k-nearest neighbors
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input"],
)

# Chain
chain = LLMChain(
    prompt=prompt,
    llm=chat_llm,
    verbose=True,
)

# QA Prompts
query = "Halo, ini dengan Ghif"
print(f"query: {query}")
response = chain.predict(input=query)

query = "Gimana caranya daftar di progam Magang dan Studi Independent Bersertifikat (MBKM)?"
print(f"query: {query}")
response = chain.predict(input=query)

query = "Tadi saya tanya tentang daftar ke program apa?"
print(f"query: {query}")
response = chain.predict(input=query)