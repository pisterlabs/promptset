from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv, find_dotenv
import openai

from langchain.vectorstores import Chroma
from inputPage.constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY


embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME)

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
def get_result(string):
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    template = """
    You are an helpful AI model that checks for user compliance, system privileges and rule violation in audit logs.You are given rules and context. Check if any rule is violated  in the context
    IMPORTANT DO NOT ANSWER WITH "As an AI model..." anytime 
    IMPORTANT when you find a violation, quote it and tell how it can be fixed 
    Go line by line and check for violations. Make sure you do not miss a violation if there is one. 
    Use the following context (delimited by <ctx></ctx>), rules (delimited by <rule></rule>) the chat history (delimited by <hs></hs>):
    ------
    <rule>
    {question}
    </rule>
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    Violations:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        }
    )



    result = qa.run(string)
    print(result)
    print(PERSIST_DIRECTORY)
    return result


