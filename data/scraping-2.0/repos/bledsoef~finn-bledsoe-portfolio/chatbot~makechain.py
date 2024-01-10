from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate

import dotenv
from pathlib import Path
import os


env_path = Path('.') / '.env'
dotenv.load_dotenv()

condense_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

qa_prompt = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say the words "I don't know". DO NOT try to make up an answer or say anything else.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    {context}

Question: {question}
Helpful answer in markdown:"""

QA_PROMPT = PromptTemplate(template=qa_prompt, input_variables=['context', 'question'])
CONDENSE_PROMPT = PromptTemplate(template=condense_prompt, input_variables=('chat_history', 'question'))

def makeChain(vectorStore:Pinecone):
    model = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=os.environ["OPENAI_API_KEY"])
    chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=vectorStore.as_retriever(), condense_question_prompt=CONDENSE_PROMPT, qa_prompt=QA_PROMPT)
    return chain