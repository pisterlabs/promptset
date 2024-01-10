#!/usr/bin/env python3
import os
import time
import json

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers, OpenAI
from langchain.chat_models import ChatOpenAI

import openai

from memory import *
from embeddings import *
from vector_stores import *
from constants import *

# from cache import *

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
model_path = os.environ.get("MODEL_PATH")
# target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 10))
emb_type = os.environ.get("EMB_TYPE")
vs_type = os.environ.get("VS_TYPE")
search_kwargs = SEARCH_KWARGS


def get_retriever(vs_type):
    if vs_type == "redis":
        db = load_redis_vector_store()
        retriever = db.as_retriever(K=10, score_threshold=0.9)
    elif vs_type == "chroma":
        db = load_chroma_vector_store(persist_directory)
        retriever = db.as_retriever(search_kwargs=search_kwargs)
    return retriever


def get_qa_chain(llm, retriever, memory):
    """
    Get the QA chain.
    """
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        # condense_question_llm=cond_llm,
        chain_type=CHAIN_TYPE,
        return_source_documents=True,
    )
    # qa.combine_docs_chain.llm_chain.prompt = QNA_PROMPT
    return qa


def main(vs_type):
    # Prepare the retriever
    if vs_type == "redis":
        db = load_redis_vector_store()
        retriever = db.as_retriever(K=10, score_threshold=0.9)
    else:
        retriever = get_retriever(vs_type)
    memory = load_chat_history(qa=True, vs_type=vs_type)
    # Prepare the QA chain
    qa = get_qa_chain(llm, retriever, memory)
    # Interactive questions and answers over your docs
    while True:
        query = input("\nEnter a question: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        try:
            print("Thinking... Please note that this can take a few minutes.")
            start = time.time()
            res = qa({"question": query})
            answer, docs = res["answer"], res["source_documents"]
            end = time.time()

            # # Print the result
            # print("\n\n> Question:")
            # print(query)
            print(f"\n> Answer (took {round(end - start, 2)} s.):")
            # printmd(answer)
            # Print the relevant sources used for the answer
            for document in docs:
                # print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
        except Exception as e:
            print(str(e))
            raise


def load_model(emb_type):
    try:
        # check if the model is already downloaded
        print("Loading model...")
        global llm, cond_llm
        # initialize llm
        emb_type = bool(emb_type == "hf")
        if emb_type:
            llm = CTransformers(
                # model="TheBloke/Llama-2-7B-Chat-GGML",
                model=os.path.abspath(model_path),
                model_type="llama",
                callbacks=[StreamingStdOutCallbackHandler()],
                config={"temperature": 0.1},
            )
        else:
            llm = ChatOpenAI(
                model_name=CHAT_MODEL_16K,
                temperature=0.1,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
            # cond_llm = OpenAI(model=CONDENSE_MODEL, temperature=0.1)
        return True

    except Exception as e:
        print(str(e))
        raise


if __name__ == "__main__":
    # load model if it has already been downloaded. If not prompt the user to download it.
    load_model(emb_type)
    main(vs_type)
