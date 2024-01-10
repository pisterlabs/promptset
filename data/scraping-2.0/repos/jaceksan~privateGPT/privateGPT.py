#!/usr/bin/env python3
import argparse
import os
from time import time
import psutil

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.vectorstores import Chroma

from constants import CHROMA_SETTINGS

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")

model_type = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))


def main():
    # Parse the command line arguments
    args = parse_arguments()
    prefix_prompt = ""
    if args.role:
        with open(f"prompts/{args.role}.txt") as fp:
            prefix_prompt = fp.read()
            if args.role == "gooddata":
                prefix_prompt = prefix_prompt.format(workspace_id=args.workspace_id)

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(
                model_path=model_path,
                n_ctx=model_n_ctx,
                callbacks=callbacks,
                verbose=False,
                n_threads=args.parallelism,
            )
        case "GPT4All":
            llm = GPT4All(
                model=model_path,
                n_ctx=model_n_ctx,
                backend="gptj",
                callbacks=callbacks,
                verbose=False,
                n_threads=args.parallelism,
            )
        case _:
            raise Exception(f"Model {model_type} not supported!")

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source
    )
    # Interactive questions and answers
    while True:
        try:
            query = input("\nEnter a query: ")
            if query == "exit":
                break
            elif not query.strip():
                continue

            # Get the answer from the chain
            start = time()
            res = qa(f"{prefix_prompt}\nQuestion:\n{query}")
            answer, docs = res["result"], [] if args.hide_source else res["source_documents"]

            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)
            duration = int((time() - start) * 1000)
            print(f"\n#####################################################################################")
            print(f"\n> Duration: {duration} ms")
            print(f"\n#####################################################################################")

            # Print the relevant sources used for the answer
            if not args.hide_source:
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
        except EOFError or KeyboardInterrupt:
            print("Ending ...")
            break


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="privateGPT: Ask questions to your documents without an internet connection,"
        " using the power of LLMs."
    )
    parser.add_argument(
        "--hide-source",
        "-S",
        action="store_true",
        help="Use this flag to disable printing of source documents used for answers.",
    )
    parser.add_argument(
        "--mute-stream",
        "-M",
        action="store_true",
        help="Use this flag to disable the streaming StdOut callback for LLMs.",
    )
    parser.add_argument(
        "-p", "--parallelism", type=int,
        help=f"How many threads can be used? Default={psutil.cpu_count(logical=False)}",
        default=psutil.cpu_count(logical=False)
    )
    parser.add_argument(
        "-w", "--workspace-id",
        help=f"GoodData workspace ID. All answers will be generated in the context of this workspace",
        default="demo"
    )
    parser.add_argument(
        "-r", "--role",
        help="If specified, file prompts/<role>.txt is injected into each question as prompt.",
        default=None
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
