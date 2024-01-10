from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
import pinecone
import json

load_dotenv()

################################################
#  create llm
################################################
# llm = ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(
    temperature=0.7, model_name="gpt-4-1106-preview"
)

################################################
#  vectore store
################################################
# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

embeddings = OpenAIEmbeddings()

# pinecone_index = pinecone.Index('auditme')
pinecone_index = Pinecone.from_existing_index('auditme', embeddings)

retriever = pinecone_index.as_retriever()

################################################
#  pipeline
################################################


# Prompt
template = """
You are an AI Smart Contract auditor agent in a RAG system. 
We have performed a vector search of known smart contract vulnerabilities based on the code in the USER QUESTION.
The results are below:

RELEVANT_VULNERNABILITY: {context}

With this knowledge, Review the following smart contract code in USER QUESTION in detail and very thoroughly.
ONLY indentify vulnerabilities in the USER QUESTION, do not analyze the RELEVANT_VULNERNABILITY.

Think step by step, carefully. 
Is the following smart contract vulnerable to '{vulnerability_type}' attacks? 
Reply with YES or NO only. Do not be verbose. 
Think carefully but only answer with YES or NO! To help you, find here a definition of a '{vulnerability_type}' attack: {vulnerability_description}

USER QUESTION: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Docs
question = "smart contract code here"
docs = retriever.get_relevant_documents(question)

# Chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)

# Run
output = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
print(output)


