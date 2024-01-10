# Memory + Sources using ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import config
import logging
import credentials
import streamlit as st
import os

OPEN_API_KEY = credentials.OPEN_API_KEY

# Initialize logging with the specified configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)


# Define answer generation function
def answer(prompt: str) -> str:

    # Log a message indicating that the function has started
    LOGGER.info(f"Start answering based on prompt: {prompt}.")

    # load persisted database from disk, and use it as normal
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)
    db = Chroma(persist_directory=config.PERSIST_DIR, embedding_function=embeddings)

    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])

    # Initiate retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": config.k})

      # test prompts for conversational Retrieval Chain
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=_template
    )
    QA_PROMPT = prompt_template

    # Create memory object to track input and outputs to hold a conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    # Initialise Conversational Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(openai_api_key=OPEN_API_KEY, model_name="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        qa_prompt=QA_PROMPT,
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )

    # Log a message indicating the number of chunks to be considered when answering the user's query.
    LOGGER.info(f"The top {config.k} chunks are considered to answer the user's query.")

    # Call the QA object to generate an answer to the prompt.
    result = qa({"question": prompt})
    answer = result['answer']

    # Log a message indicating the answer that was generated
    LOGGER.info(f"The returned answer is: {answer}")

    # Log a message indicating that the function has finished and return the answer.
    LOGGER.info(f"Answering module over.")

    return answer
