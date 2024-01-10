from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
import pickle

template = """
    Given the following Chat History and a Follow-Up Question, rephrase the Follow-Up Question to be a Stand-Alone Question.
    You can assume the Follow-Up Question is about finding a creative professional from your team.

    Chat History:
    {chat_history}
    Follow-Up Question:
    {question}
    Stand-Alone Question:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

qa_template = """
    You are an AI assistant for answering questions about a team of creative professionals.
    You are given the following extracted parts of your creative roster, and a question. Provide a conversational answer.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
    If the question is not about your team of creatives, politely inform them that you are tuned to only answer questions about the creatives on your team.
    Question:
    {question}
    =========
    Profiles:
    {context}
    =========
    Answer in Markdown:
"""
QA_PROMPT = PromptTemplate(
    template=qa_template, input_variables=["question", "context"]
)


def load_retriever():
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_custom_prompt_qa_chain():
    llm = ChatOpenAI(temperature=1)
    retriever = load_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/6635
    # see: https://github.com/langchain-ai/langchain/issues/1497
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return model


def get_condense_prompt_qa_chain():
    llm = ChatOpenAI(temperature=1)
    retriever = load_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/5890
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return model


chain_options = {
    "custom_prompt": get_custom_prompt_qa_chain,
    "condense_prompt": get_condense_prompt_qa_chain,
}
