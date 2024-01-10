import os
from time import sleep

from gpt_index import (
    GPTListIndex,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    SimpleDirectoryReader,
    readers,
)
from langchain import OpenAI

# Enter your API Key Here:
API_KEY = "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"
os.environ["OPENAI_API_KEY"] = API_KEY

# Enter where your data is Here:
DOCUMENTS_FOLDER = "/Users/praburajasekaran/gpt/sample/"


def wait_secs(remaining_time: int) -> None:
    """Waits x seconds."""
    while remaining_time != 0:
        print(f"Please Wait for {remaining_time}", end="\r")
        remaining_time -= 1
        sleep(1)


def construct_index():
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 1000

    # define LLM
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0.7,
            model_name="text-davinci-003",
            max_tokens=num_outputs,
        )
    )
    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit
    )

    documents = SimpleDirectoryReader(DOCUMENTS_FOLDER).load_data()

    index = GPTSimpleVectorIndex([], llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    # index each document indiviually so we can reindex it when we reach limit exception.
    for document in documents:
        try:
            index.insert(document)
        except Exception:
            print("Max Indexing Limit")
            wait_secs(60)
            index.insert(document)

    index.save_to_disk("index.json")


def ask_the_journal():
    index = GPTSimpleVectorIndex.load_from_disk("index.json")

    RETRIEVE_DOCUMENTS = 0
    SUMMARIZE_DOCUMENTS = 1

    state = RETRIEVE_DOCUMENTS

    while True:
        if state == RETRIEVE_DOCUMENTS:
            query = input("What kinds of documents do you want to extract? ")

            try:
                response = index.query(
                    query, similarity_top_k=50, verbose=True, response_mode="no_text"
                )
            except Exception:
                print("Max Limit Reached.")
                wait_secs(60)
                response = index.query(
                    query, similarity_top_k=50, verbose=True, response_mode="no_text"
                )

            retrieved_docs = []
            for sn in response.source_nodes:
                retrieved_docs.append(readers.Document(sn.source_text))

            state = SUMMARIZE_DOCUMENTS

            index2 = GPTListIndex([])
            for doc in retrieved_docs:
                try:
                    index2.insert(doc)
                except Exception:
                    print("Max Limit Reached.")
                    wait_secs(60)
                    index2.insert(doc)

            while state == SUMMARIZE_DOCUMENTS:
                query = input("What would you like to know? ")

                if query == "exit":
                    state = RETRIEVE_DOCUMENTS
                    break

                try:
                    response = index2.query(query, response_mode="compact", verbose=True)
                except Exception:
                    print("Max Limit Reached.")
                    wait_secs(60)
                    response = index2.query(query, response_mode="compact", verbose=True)

                print(response.response)


if __name__ == "__main__":
    generate_new_index = input("Do you want to generate a new index.json? (Y/N) ").strip().lower()
    if generate_new_index == "y":
        construct_index()
    ask_the_journal()
