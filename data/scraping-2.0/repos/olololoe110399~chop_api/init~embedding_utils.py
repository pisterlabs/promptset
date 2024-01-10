import json
import os
from concurrent.futures import ThreadPoolExecutor

from langchain.embeddings import OpenAIEmbeddings


def get_embeddings(text: str, model="text-embedding-ada-002", chunk_size=1000) -> list[float]:
    embeddings_model = OpenAIEmbeddings(model=model)
    # Get embeddings for the text
    res = embeddings_model.embed_documents([text], chunk_size=chunk_size)[0]
    # Return the embeddings
    return res


def embed_chunk(i, text):
    # Get the embeddings for the text
    return i, get_embeddings(text)


def embed_documents_in_json_file(input_file_path, output_file_path):
    with open(input_file_path, "r") as file:
        data = json.load(file)['data']

    # Embed the documents
    with ThreadPoolExecutor() as executor:
        # List to hold the future objects
        futures = []

        # Iterate through the data, submitting each JSON chunk to the executor
        for i, item in enumerate(data):
            if "embeddings" not in item.keys():
                json_object = json.dumps(item)
                future = executor.submit(embed_chunk, i, json_object)
                futures.append(future)

        # Retrieve the results from the futures, keeping the order
        for future in futures:
            i, embeddings = future.result()
            data[i]["embeddings"] = embeddings

    # Save the output file
    with open(output_file_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    input_file_path = os.path.join(os.getcwd(), "resources/data.json")
    output_file_path = os.path.join(os.getcwd(), "resources/processed_data_with_embeddings.json")
    embed_documents_in_json_file(input_file_path, output_file_path)
