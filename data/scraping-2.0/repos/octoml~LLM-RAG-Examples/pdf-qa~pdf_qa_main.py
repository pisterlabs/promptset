import logging
import os
import sys

from pathlib import Path
from dotenv import load_dotenv
from langchain.llms.octoai_endpoint import OctoAIEndpoint as OctoAiCloudLLM
from langchain.embeddings.octoai_embeddings import OctoAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import time
import shutil
from termios import tcflush, TCIFLUSH

# Set up logging
logging.basicConfig(level=logging.CRITICAL)

# Load environment variables
load_dotenv()

# Define the file storage directory
FILES_DIR = Path("files")


def init_files_directory():
    """
    Initialize the files directory.
    """
    FILES_DIR.mkdir(exist_ok=True)


def handle_exit():
    """
    Handle exit gracefully.
    """
    print("\nGoodbye!\n")
    sys.exit(0)


def clear_screen():
    """
    Clear the terminal screen.
    """

    term_size = shutil.get_terminal_size((80, 20))
    print("\n" * term_size.lines, end="")
    sys.stdout.flush()


def extract_text_from_pdf(pdf_path):
    """
    Extract text from the given PDF file.
    """
    pdf_reader = PdfReader(pdf_path)
    return "".join(page.extract_text() or "" for page in pdf_reader.pages)


def setup_langchain_environment():
    """
    Set up the language model and embeddings.
    """
    endpoint_url = os.getenv("ENDPOINT_URL")
    if not endpoint_url:
        raise ValueError("The ENDPOINT_URL environment variable is not set.")

    # Initialize the LLM and Embeddings
    llm = OctoAiCloudLLM(
        endpoint_url=endpoint_url,
        model_kwargs={
            "model": "llama-2-70b-chat-fp16",
            "messages": [
                {
                    "role": "system",
                    "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                }
            ],
            "stream": False,
            "max_tokens": 256,
        },
    )
    embeddings = OctoAIEmbeddings(
        endpoint_url="https://instructor-large-f1kzsig6xes9.octoai.run/predict"
    )
    return llm, embeddings


def interactive_qa_session(file_path):
    """
    Interactively answer user questions about the document.
    """
    print("Loading...")
    raw_text = extract_text_from_pdf(file_path)
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=400, chunk_overlap=100, length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    llm, embeddings = setup_langchain_environment()
    print("Creating embeddings")
    document_search = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(llm, chain_type="stuff")

    clear_screen()
    print("Ready! Ask anything about the document.")
    print("\nPress Ctrl+C to exit.")

    try:
        from termios import tcflush, TCIFLUSH
        tcflush(sys.stdin, TCIFLUSH)
        while True:
            prompt = input("\nPrompt: ").strip()
            if not prompt:
                continue
            if prompt.lower() == "exit":
                handle_exit()

            start_time = time.time()
            docs = document_search.similarity_search(prompt)
            response = chain.run(input_documents=docs, question=prompt)
            elapsed_time = time.time() - start_time
            print(f"Response ({round(elapsed_time, 1)} sec): {response}\n")
    except KeyboardInterrupt:
        handle_exit()


def select_file():
    """
    Select a file for processing.
    """
    os.system("clear")
    files = [file for file in os.listdir(FILES_DIR) if file.endswith(".pdf")]

    if not files:
        return "file.pdf" if os.path.exists("files/file.pdf") else None

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
            file_path = os.path.abspath(os.path.join(FILES_DIR, files[selection - 1]))

        return file_path
    except ValueError:
        return select_file()


if __name__ == "__main__":
    init_files_directory()
    selected_file = select_file()
    if selected_file:
        interactive_qa_session(selected_file)
    else:
        print("No files found.")
        handle_exit()
