import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain.llms.octoai_endpoint import OctoAIEndpoint as OctoAiCloudLLM
from langchain.embeddings.octoai_embeddings import OctoAIEmbeddings
from llama_index import (
    LLMPredictor,
    ServiceContext,
    download_loader,
    GPTVectorStoreIndex,
    LangchainEmbedding,
)

import time
from termios import tcflush, TCIFLUSH

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(current_dir)
# Set logging level to CRITICAL
logging.basicConfig(level=logging.CRITICAL)

# Load environment variables
load_dotenv()

# Set the file storage directory
FILES = "./files"


def init():
    """
    Initialize the files directory.
    """
    if not os.path.exists(FILES):
        os.mkdir(FILES)


def handle_exit():
    """
    Handle exit gracefully.
    """
    print("\nGoodbye!\n")
    sys.exit(1)


def ask(file):
    """
    Load the file, create the query engine and interactively answer user questions about the document.
    """
    print("Loading...")
    # Load the PDFReader
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=Path(file))

    # Initialize the OctoAiCloudLLM
    endpoint_url = os.getenv("ENDPOINT_URL")
    # Set up the language model and predictor
    llm = OctoAiCloudLLM(
        endpoint_url=endpoint_url,
        model_kwargs={
        "model": "llama-2-7b-chat",
        "messages": [
            {
                "role": "system",
                "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            }
        ],
        "stream": False,
        "max_tokens": 256
        }
    )


    llm_predictor = LLMPredictor(llm=llm)

    # Create the LangchainEmbedding
    embeddings = LangchainEmbedding(
        OctoAIEmbeddings(
            endpoint_url="https://instructor-large-f1kzsig6xes9.octoai.run/predict"
        )
    )

    # Create the ServiceContext
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=512, embed_model=embeddings
    )

    # Create the index from documents
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

    # Create the query engine
    query_engine = index.as_query_engine(verbose=True, llm_predictor=llm_predictor)

    # Clear the screen
    os.system("clear")

    print("Ready! Ask anything about the document")
    print("")
    print("Press Ctrl+C to exit")

    try:
        tcflush(sys.stdin, TCIFLUSH)
        while True:
            prompt = input("\nPrompt: ")
            if prompt is None:
                continue
            if prompt == "exit":
                handle_exit()

            start_time = time.time()
            response = query_engine.query(prompt)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print()

            # Transform response to string and remove leading newline character if present
            response = str(response).lstrip("\n")

            print(f"Response({round(elapsed_time, 1)} sec): {response}")
    except KeyboardInterrupt:
        handle_exit()


def select_file():
    """
    Select a file for processing.
    """
    os.system("clear")
    files = [file for file in os.listdir(FILES) if file.endswith(".pdf")]

    if not files:
        return "file.pdf" if os.path.exists("file.pdf") else None

    print("Select a file")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")
    print()

    try:
        possible_selections = list(range(len(files) + 1))
        selection = int(input("Enter a number, or 0 to exit: "))

        if selection == 0:
            handle_exit()
        elif selection not in possible_selections:
            select_file()
        else:
            file_path = os.path.abspath(os.path.join(FILES, files[selection - 1]))

        return file_path
    except ValueError:
        return select_file()


if __name__ == "__main__":
    # Initialize the file directory
    init()
    if file := select_file():
        # Start the interactive query session
        ask(file)
    else:
        print("No files found")
        handle_exit()
