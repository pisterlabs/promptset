from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.prompts import PromptTemplate

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain.chains import VectorDBQAWithSourcesChain


def load_chat_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    return chain


def load_context_qa_chain(raw_documents):
    text_splitter = NLTKTextSplitter(chunk_size=1000)
    documents = text_splitter.split_documents(raw_documents)
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

    _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from the given context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Summary of conversation:
    {history}
    Context
    {summaries}
    Human: {question}
    AI:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "summaries", "question"], template=_DEFAULT_TEMPLATE
    )

    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=vectorstore,
                                                combine_prompt=PROMPT)
    return chain

def load_yt_chain(original_docs):
    docs = []
    metadatas = []
    for i, d in enumerate(original_docs):
        metadatas.extend([{"source": {"start": d["start"], "end": d["end"]}}])
        docs.append(d["text"])
    # Here we create a vector store from the documents and save it to disk.
    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)

    _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from the given context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Summary of conversation:
        {history}
        Context
        {summaries}
        Human: {question}
        AI:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "summaries", "question"], template=_DEFAULT_TEMPLATE
    )

    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
    return chain

