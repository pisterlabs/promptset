#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time

load_dotenv()

# MODEL_TYPE: 
# either "LlamaCpp" or "GPT4All"
# PERSIST_DIRECTORY: 
# Provide the absolute path to the desired directory.
# LLAMA_EMBEDDINGS_MODEL: 
# absolute path of your LlamaCpp-supported embeddings model binary. 
# MODEL_PATH: 
# path of your GPT4All or LlamaCpp-supported LLM model
# MODEL_N_CTX: 
# Specify the maximum token limit for both the embeddings and LLM models.

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
print("embeddings_model_name=", embeddings_model_name)

persist_directory = os.environ.get('PERSIST_DIRECTORY')
print("persist_directory=", persist_directory)

model_type = os.environ.get('MODEL_TYPE')
print("model_type=", model_type)

model_path = os.environ.get('MODEL_PATH')
print("model_path=", model_path)

model_n_ctx = os.environ.get('MODEL_N_CTX')
print("model_n_ctx=", model_n_ctx)

model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
print("model_n_batch=", model_n_batch)

target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
print("target_source_chunks=", target_source_chunks)

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            #llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
 
            llm = GPT4All(model=model_path,  verbose=True)

        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break        
        if query == "quit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        prompt = "Assume you are a professor of science and you are explaining concept to a computer science student. " + query
        res = qa(prompt)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n  Question:\n", query)
        print(f"\n\n  ANSWER took {round(end - start, 2)} seconds:\n", answer)

        # Print the relevant sources used for the answer
        print("\n\n  DOCUMENT SOURCES: \n" )
        count = 1
        for document in docs:
            print("\n\n    ", count, "). \"" + document.metadata["source"] + "\":")
            print("\n    EXCERPT: \n    " + document.page_content)
            count = count + 1

def parse_arguments():
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
