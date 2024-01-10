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

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))


# convert to integer if not None
if model_n_batch is not None:
    model_n_batch = int(model_n_batch)
if target_source_chunks is not None:
    target_source_chunks = int(target_source_chunks)

variables = [embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, model_n_batch, target_source_chunks]

for var in variables:
    if var is None:
        raise EnvironmentError("One or more environment variables are None.")

# from constants import CHROMA_SETTINGS
from privateGPT.constants import CHROMA_SETTINGS

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
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Open 'data.txt' file in append mode
    with open('data.txt', 'a') as f:
        # Interactive questions and answers
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break
            if query.strip() == "":
                continue

            # Get the answer from the chain
            start = time.time()
            res = qa(query)
            answer, docs = res['result'], [] if args.hide_source else res['source_documents']
            end = time.time()

            # Write the input and output to the file
            f.write("\n\n> Question:\n")
            f.write(query)
            f.write(f"\n\n> Answer (took {round(end - start, 2)} s.):\n")
            f.write(answer)

            # Print the result
            print("\n\n> Question:")
            print(query)
            print(f"\n> Answer (took {round(end - start, 2)} s.):")
            print(answer)

            # Write the relevant sources used for the answer to the file
            for document in docs:
                f.write("\n> " + document.metadata["source"] + ":\n")
                f.write(document.page_content)
                # Print the relevant sources used for the answer
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)


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
