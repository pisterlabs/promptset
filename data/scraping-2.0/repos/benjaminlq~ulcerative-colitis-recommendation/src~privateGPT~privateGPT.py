"""CLI Interactive App
"""
import argparse

from chromadb.config import Settings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.vectorstores import Chroma

from config import (
    EMBSTORE_DICT,
    PGPT_EMBEDDINGS_MODEL,
    PGPT_MODEL,
    PGPT_MODEL_N_CTX,
    PGPT_MODEL_TYPE,
)


def get_argument_parser():
    """Argument Parser

    Returns:
        args: argument dictionary
    """
    parser = argparse.ArgumentParser("Embedding Store Creation")
    parser.add_argument(
        "--embed_store", "-e", type=str, default="chroma", help="chroma|faiss|pinecone"
    )
    parser.add_argument(
        "--embed_path",
        "-p",
        type=str,
        default=EMBSTORE_DICT["chroma"],
        help="output directory to store embeddings",
    )
    parser.add_argument(
        "--emb_model",
        "-em",
        type=str,
        default=PGPT_EMBEDDINGS_MODEL,
        help="path to embedding model",
    )
    parser.add_argument(
        "--llm_model",
        "-lm",
        type=str,
        default=PGPT_MODEL,
        help="path to embedding model",
    )
    parser.add_argument(
        "--llm_type",
        "-t",
        type=str,
        default=PGPT_MODEL_TYPE,
        help="path to embedding model",
    )
    parser.add_argument(
        "--token_count", "-c", type=int, default=PGPT_MODEL_N_CTX, help="token count"
    )
    args = parser.parse_args()
    return args


def main():
    """Main Function"""
    args = get_argument_parser()

    emb_store_type = args.embed_store.lower()
    embeddings_model_name = args.emb_model
    emb_directory = args.embed_path

    model_type = args.llm_type
    model_path = args.llm_model
    model_n_ctx = args.token_count

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    chroma_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=emb_directory,
        anonymized_telemetry=False,
    )

    if emb_store_type == "chroma":
        db = Chroma(
            persist_directory=emb_directory,
            embedding_function=embeddings,
            client_settings=chroma_settings,
        )
    retriever = db.as_retriever()

    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    # match model_type:
    #     case "LlamaCpp":
    #         llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
    #     case "GPT4All":
    #         llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    #     case _default:
    #         print(f"Model {model_type} not supported!")
    #         exit;

    if model_type.lower() == "llamaccp":
        llm = LlamaCpp(
            model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False
        )
    elif model_type.lower() == "gpt4all":
        llm = GPT4All(
            model=model_path,
            n_ctx=model_n_ctx,
            backend="gptj",
            callbacks=callbacks,
            verbose=False,
        )
    else:
        raise Exception(f"Model {model_type} not supported!")

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


if __name__ == "__main__":
    main()

# python3 ./src/privateGPT/privateGPT.py
