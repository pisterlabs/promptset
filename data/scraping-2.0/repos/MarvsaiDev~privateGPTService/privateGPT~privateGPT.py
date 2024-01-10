#!/usr/bin/env python3
from typing import List

from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

import privateGPT.global_vars as constants
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
import openai
from langchain.llms import GPT4All, LlamaCpp, AzureOpenAI, OpenAIChat
import chromadb
import os
import argparse
import time
import logging as log
openai.api_type = "azure"

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)
openai.api_base = os.environ['OPENAI_API_BASE']
openai.api_version = os.environ['OPENAI_API_VERSION']
openai.api_key = os.environ['OPENAI_API_KEY']
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_subtype = os.environ.get('MODEL_SUBTYPE', 'gpt-35-16k' )
model_path = os.environ.get('MODEL_PATH') #model name
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

CHROMA_SETTINGS = constants.CHROMA_SETTINGS
qa_system = None
def main(commandLine=True, persistDir=None, lmodel_type=model_type, numpages = 10)->BaseRetrievalQA:
    # Parse the command line arguments
    # args = parse_arguments()
    global persist_directory
    if not persistDir:
        persistDir = persist_directory
    if '-ada-' in embeddings_model_name:
        embeddings = OpenAIEmbeddings(deployment=embeddings_model_name, engine=embeddings_model_name)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persistDir)
    nofiles = os.listdir(persistDir)
    no_of_pdfs = sum(['pdf' in nof for nof in nofiles])
    db = Chroma(persist_directory=persistDir, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_type='similarity_score_threshold',search_kwargs={'k':numpages,'score_threshold':0.0001}) #search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = []

    # Prepare the LLM
    dengine = 'gpt-35-16k'
    match lmodel_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case 'OpenAIChat':
            llm = ChatOpenAI(model_name=model_subtype, max_tokens = 4000 if '16k' in dengine else 2000, temperature=0.0,
                model_kwargs=dict(engine=dengine,top_p=0.01))
        case _default:
            llm = AzureOpenAI(
                deployment_name='text-davinci-003',
                model_name="claritus003",
                max_tokens = 1800,
                top_p=0.01,
                temperature= 0
            )

    from langchain.prompts import PromptTemplate
    prompt_template = """Use the following pieces of context to answer the question at the end.
        '{context}'
        Question: {question}
        """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
    # Interactive questions and answers
    global qa_system
    qa_system = qa
    while commandLine:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], []
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
    return qa

def func_print(srcdocs:List[Document] ):
    if not isinstance(srcdocs, list):
        print(srcdocs)
        return
    for x in srcdocs:
        print(x.page_content)
def answer_query(query, jobid=None, qs=None, update_callback=None, metadata:dict=None):

    if qs:
        _qa_system = main(False, jobid)
    else:
        if not jobid:
            raise Exception('answer_query 118: jobid must be specifed or Query system')
        _qa_system = main(False, jobid)

    res = _qa_system(query, metadata=metadata if metadata else {})
    answer, docs = res['result'], res['source_documents']
    func_print(res['source_documents'])
    log.info(res)

    if update_callback:
        update_callback(answer)
    else:
        return answer, docs, _qa_system
def parse_arguments():

    # question:
    # what is last invoice with acc po?
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
