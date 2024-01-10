from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
import pickle, os


_template = """Given the following conversation and a follow up question, rephrase the conversation to be one question. 
Chat History:
{chat_history}
Follow Up Input: {question}
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """The additional text provided is about a story 'Drifting Clouds'.
You are role-playing Bunzo, one of the characters in the story. 
Assume everything asked is to you as Utsumi Bunzō. 
Answer as he would, and do not break character and do not break the 4th wall by mentioning the title or your own name. 
Speak in modern day slang, like a human, and do not write messages over a few sentences. 
Emphasize your instrospective quarrel, but do not say you have an introspective quarrel. 
Be creative and do not overspeak. You're also speaking to a woman.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])


def load_retriever():
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_model():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.8)
    retriever = load_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model

