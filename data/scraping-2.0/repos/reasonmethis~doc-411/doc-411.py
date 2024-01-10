import sys
import os
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain

from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT_CHAT

# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

import docgrab

USE_SOURCES = False
VERBOSE = True
DELIMITER = "-" * 94 + "\n"
INTRO_ASCII_ART = """ ,___,   ,___,   ,___,                                                 ,___,   ,___,   ,___,
 [OvO]   [OvO]   [OvO]                                                 [OvO]   [OvO]   [OvO]
 /)__)   /)__)   /)__)               WELCOME TO DOC 411                /)__)   /)__)   /)__)
--"--"----"--"----"--"--------------------------------------------------"--"----"--"----"--"--"""


def create_vectorstore(docs, save_dir=None):
    try:
        # index = VectorstoreIndexCreator().from_loaders([loader])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        docs = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            docs, embeddings, persist_directory=save_dir
        )
        return vectorstore
    except Exception as e:
        # print authentication errors etc.
        print(e)
        sys.exit()


def create_bot(vectorstore):
    try:
        if "gpt" in MODEL_NAME or "text-davinci" in MODEL_NAME:
            llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)  # for answering
            llm_q = ChatOpenAI(model=MODEL_NAME, temperature=0)  # to condense question
        else:
            llm = OpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
            llm_q = OpenAI(model=MODEL_NAME, temperature=0)

        # https://python.langchain.com/docs/modules/chains/popular/chat_vector_db
        combine_docs_chain = (
            load_qa_with_sources_chain(llm, verbose=VERBOSE)
            if USE_SOURCES
            else load_qa_chain(llm, prompt=QA_PROMPT_CHAT, verbose=VERBOSE)
        )
        bot = ConversationalRetrievalChain(
            question_generator=LLMChain(
                llm=llm_q, prompt=CONDENSE_QUESTION_PROMPT, verbose=VERBOSE
            ),
            retriever=vectorstore.as_retriever(),
            combine_docs_chain=combine_docs_chain,
            return_source_documents=True,
            return_generated_question=True,
        )
        return bot
    except Exception as e:
        print(e)
        sys.exit()


if __name__ == "__main__":
    print(INTRO_ASCII_ART + "\n\n")

    # check that the necessary environment variables are set
    load_dotenv()
    VECTORDB_DIR = os.getenv("VECTORDB_DIR")
    DOCS_TO_INGEST_DIR_OR_FILE = os.getenv("DOCS_TO_INGEST_DIR_OR_FILE")
    SAVE_VECTORDB_DIR = os.getenv("SAVE_VECTORDB_DIR")

    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
    if (not DOCS_TO_INGEST_DIR_OR_FILE and not VECTORDB_DIR) or not os.getenv(
        "OPENAI_API_KEY"
    ):
        print("Please set the environment variables in .env, as shown in .env.example.")
        sys.exit()

    # verify the validity of the docs or db path
    tmp_path = VECTORDB_DIR or DOCS_TO_INGEST_DIR_OR_FILE
    if not os.path.exists(tmp_path):
        print(
            "The path you provided for your documents or vector database does not exist."
        )
        sys.exit()

    # load documents to ingest (if we're not loading a vectorstore)
    if VECTORDB_DIR:
        print("Loading the vector database of your documents... ", end="", flush=True)
    else:
        print("Ingesting your documents, please stand by... ", end="", flush=True)
        if os.path.isfile(DOCS_TO_INGEST_DIR_OR_FILE):
            if DOCS_TO_INGEST_DIR_OR_FILE.endswith(".jsonl"):
                loader = docgrab.JSONLDocumentLoader(DOCS_TO_INGEST_DIR_OR_FILE)
            else:
                loader = TextLoader(DOCS_TO_INGEST_DIR_OR_FILE)
        else:
            loader = DirectoryLoader(DOCS_TO_INGEST_DIR_OR_FILE)
        docs = loader.load()

    # create (or load) vectorstore and bot
    if VECTORDB_DIR:
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(), persist_directory=VECTORDB_DIR
        )
    else:
        vectorstore = create_vectorstore(docs, save_dir=SAVE_VECTORDB_DIR or None)
    bot = create_bot(vectorstore)
    print("Done!")

    # start chat
    print()
    print("Keep in mind:")
    print("- Replies may take a few seconds.")
    print("- Doc 411 remembers the previous messages but not always accurately.")
    print('- To exit, type "exit" or "quit", or just press Enter twice.')
    print(DELIMITER)
    chat_history = []
    while True:
        # get query from user
        query = input("YOU: ")
        if query == "exit" or query == "quit":
            break
        if query == "":
            print("Please enter your query or press Enter to exit.")
            query = input("YOU: ")
            if query == "":
                break
        print()

        # get response from bot
        # reply = index.query(query, llm=ChatOpenAI() if USE_GENERAL_KNOWLEDGE else None)
        result = bot({"question": query, "chat_history": chat_history})
        reply = result["answer"]

        # update chat history
        chat_history.append((query, reply))

        # print reply
        print("DOC 411:", reply)
        print(DELIMITER)

        # print(result)
        # print(DELIMITER)
