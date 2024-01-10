import sys
from langchain.llms import OpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings

from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain


def ingest(url):
    # load data
    loader = WebBaseLoader(url)
    data = loader.load()

    # split data into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(data)

    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


def get_chain(vectorstore):
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    You can assume the question about the URL shared.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    template = """You are an AI assistant for answering questions for the job post and advice users to assess job based on the description.
    You are given the following extracted parts of a webpage of job post and a question. Provide a conversational answer.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown:"""
    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["question", "context"]
    )

    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain


if __name__ == "__main__":
    url = "https://careers.deloitte.ca/job/Toronto%2C-Ontario%2C-Canada-Lead-Data-Engineer-%28Manager%29%2C-Deloitte-Global-Technology%2C-GS-Technology-Solutions-%28Business%29-ON/975737500/"
    vectorstore = ingest(url)
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Welcome to the AI JobPost assistant!")

    while True:
        print("Human:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("AI:")
        print(result["answer"])
