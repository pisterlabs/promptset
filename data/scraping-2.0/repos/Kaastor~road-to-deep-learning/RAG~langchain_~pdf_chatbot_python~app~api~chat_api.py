import elasticapm
import os
from fastapi import APIRouter

from langchain_.pdf_chatbot_python.app.api.models import Chat
from langchain_.pdf_chatbot_python.app.service.constants import PINECONE_INDEX_NAME, PINECONE_NAME_SPACE, OPENAI_API_KEY
from langchain_.pdf_chatbot_python.app.service.data.data_ingestion import run, init_pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import (
    LLMChain, ConversationalRetrievalChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

chat_router = APIRouter()
init_pinecone()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""Test input: 
{
  "question": "give example of policy based methods",
  "history": [
    [
      "What other than value based methods you know?",
      "Based on the given context, other than value-based methods, I know about policy-based methods. Policy-based methods explicitly build a representation of a policy and keep it in memory during learning. Examples include policy gradient methods and trust-region policy optimization methods."
    ]
  ]
}

Out: Some examples of policy-based methods include REINFORCE, Actor-Critic methods, Trust Region Policy Optimization 
(TRPO), and Proximal Policy Optimization (PPO). """


@elasticapm.async_capture_span()
@chat_router.post('/chat')
async def chat(_chat: Chat):
    # elasticapm.set_custom_context({
    #     'question': _chat.question,
    #     'history': _chat.history
    # })

    # OpenAI recommends replacing newlines with spaces for best results
    vector_store = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=OpenAIEmbeddings(),
        namespace=PINECONE_NAME_SPACE
    )

    # This controls how the standalone question is generated.
    # Should take `chat_history` and `question` as input variables.
    standalone_template = (
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. \
        Chat History: {chat_history} \
        Follow Up Input: {question} \
        Standalone question:"
    )
    standalone_prompt = PromptTemplate.from_template(standalone_template)

    qa_template = (
        "You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If \
        you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question is \
        not related to the context, politely respond that you are tuned to only answer questions that are related to \
        the context. Answer in polish. Questions and responses will be connected to polish law to use polish law language \
        {context} \
        Question: {question} \
        Helpful answer in markdown: "
    )
    qa_prompt = PromptTemplate.from_template(qa_template)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    combine_docs_chain = load_qa_chain(llm=llm, prompt=qa_prompt)
    question_generator_chain = LLMChain(llm=llm, prompt=standalone_prompt)
    qa_chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(search_kwargs={"k": 8}),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator_chain,
        return_source_documents=True
    )
    result = qa_chain({"question": _chat.question.replace("\n", " "), "chat_history": _chat.history})
    return result


@elasticapm.async_capture_span()
@chat_router.post('/ingest_data')
async def ingest_data():
    run()
