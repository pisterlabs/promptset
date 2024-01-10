from langchain.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
from langchain.chains import RetrievalQA
import os

'''
Experiment with the retrieval model
'''

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['OPENAI_API_KEY']

embedding = OpenAIEmbeddings()

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_model, temperature=0)

persist_directory = './chroma_city_orlando/'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

print(vectordb._collection.count())

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

question = "what are the key aspects of the orlando budget guide?"
result = qa_chain({"query": question})

print(result["result"])


"""
#system_prompt = PromptTemplate(input_variables=['context'], template="Use the following pieces of context to answer the user's question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}")

#system2_prompt = ChatPromptTemplate( input_variables=['context', 'question'], messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], template="Use the following pieces of context to answer the user's question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}"))

#prompt2 = ChatPromptTemplate(input_variables=['context', 'question'])


messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], template="Use the following pieces of context to answer the user's \
question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}")),
HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question'], template='{question}'))]

"""


question = "what areas of information do you have for the City of Orlando?"
result = qa_chain({"query": question})

print(result["result"])