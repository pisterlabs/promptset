#! /usr/bin/env python3

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp

# set up argparse
import argparse
def init_argparse():
    parser = argparse.ArgumentParser(description='privateGPT: Use LLMS to ask questions of your documents with no internet connection')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing source documents used for answers.')
    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser

from dotenv import load_dotenv
import os
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    argparser = init_argparse()
    args = argparser.parse_args()

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        match query.split():
            case ["exit" | "quit"]:
                quit()
            case _:
            # Get the answer from the chain
                res = qa(query)
                answer, docs = res['result'], [] if args.hide_source else res['source_documents']
                # Print the result
                print(f"\n\n> Question: {query!r}")
                print("\n> Answer:")
                from pprint import pprint
                pprint(answer)

                # Print the relevant sources used for the answer
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
#                   print(document.page_content)

if __name__ == "__main__":
    main()
