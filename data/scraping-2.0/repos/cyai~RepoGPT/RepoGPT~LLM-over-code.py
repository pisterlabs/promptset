import asyncio
import time
from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()
import openai


# A async function
async def cloneRepo():
    # Clone the repo

    # write the code to check weather the repo is already cloned
    if len(os.listdir("/Users/vardh/work/trying-smt/LLM-Over-Code/repo")) != 0:
        print(os.listdir("/Users/vardh/work/trying-smt/LLM-Over-Code/repo"))
        print("Repo already cloned")
        return "/Users/vardh/work/trying-smt/LLM-Over-Code/repo"
    else:
        repo_path = "/Users/vardh/work/trying-smt/LLM-Over-Code/repo"
        repo = Repo.clone_from(
            "https://github.com/cyai/Hand_Motion_Detector", to_path=repo_path
        )   
        return repo_path


async def loadFiles(repo_path):
    # Load
    python_loader = GenericLoader.from_filesystem(
        repo_path + "/",
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )

    markdown_loader = GenericLoader.from_filesystem(
        repo_path + "/",
        glob="**/*",
        suffixes=[".md"],
        parser=LanguageParser(language=Language.MARKDOWN, parser_threshold=500),
    )

    html_loader = GenericLoader.from_filesystem(
        repo_path + "/",
        glob="**/*",
        suffixes=[".html"],
        parser=LanguageParser(language=Language.HTML, parser_threshold=500),
    )

    loader = python_loader + markdown_loader + html_loader
    documents = loader.load()
    print(len(documents))

    return documents


async def splitting(documents):
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    texts = python_splitter.split_documents(documents)
    print(len(texts))
    return texts


async def retrievalQA(texts):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print(openai.api_key)

    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
    print("Sleeping for 60 seconds...")
    time.sleep(60)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )

    return retriever


async def chat(retriever, question):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    result = qa(question)
    print(result["answer"])


async def main():
    print("Cloning the repo...")
    repo_path = await cloneRepo()

    documents = await loadFiles(repo_path)

    print("Splitting the documents...")
    texts = await splitting(documents)

    print("Retrieving the QA...")
    retriever = await retrievalQA(texts)

    print("Chatting...")
    await chat(retriever, "What is keras?")


if __name__ == "__main__":
    # print(os.getenv("OPENAI_API_KEY"))
    asyncio.run(main())


