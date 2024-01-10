from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import json
from langchain.chains import LLMChain, ConversationChain
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    SemanticSimilarityExampleSelector
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


prefix = """You are a knowledgeable customer service agent from Pusat Bantuan Merdeka Belajar Kampus Merdeka (MBKM).
Use the historical conversation and examples below to answer various questions from users.
If you don't know the answer, just say I don't know. Don't make up an answer.
The answer given must always be in Indonesian with a friendly tone.

Here are some examples:
"""

example_template = """
Human: {question}
AI Assistant: {answer}
"""

suffix = """
Human: {input}
AI Assistant:
"""

# Enable few shot example prompting -- load context examples from file
examples = json.load(open("chat_samples_nogreeting.json", "r"))


# LLM
chat_llm = ChatOpenAI(
    # model="text-davinci-002",
    model="gpt-3.5-turbo",
    verbose=True,
    temperature=0.0,
    streaming=True
)


example_prompt = PromptTemplate.from_template(template=example_template)

# Select only k number of examples in the prompt
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, 
    OpenAIEmbeddings(model="text-embedding-ada-002"),
    FAISS, 
    k=3
)

# Chain
prompt = FewShotPromptTemplate(
    prefix=prefix,
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix=suffix,
    # input_variables=["input", "chat_history"],
    input_variables=["input"],
)

chain = LLMChain(
    prompt=prompt,
    llm=chat_llm,
    # memory=memory,
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

