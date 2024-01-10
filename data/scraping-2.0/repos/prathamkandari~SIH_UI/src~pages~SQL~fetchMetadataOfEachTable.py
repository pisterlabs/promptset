from azure.storage.blob import BlobServiceClient
import yaml
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

openai.api_key = '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'

os.environ["OPENAI_API_KEY"] = "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"

def generate_summary(input_text):
            prompt = f"Summarize the following YAML data:\n\n{input_text}\n\nSummary:"
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=100  # Adjust as needed
            )
            summary = response['choices'][0]['text']
            return summary


def generate_sentence_vector(sentence):

    embedding_vector = OpenAIEmbeddings().embed_query(sentence)
    return embedding_vector


def fetch_and_parse_yaml_from_blob(connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "yamlfiles"
    container_client = blob_service_client.get_container_client(container_name)

    yaml_vector_map = {}  # Dictionary to store blob names and vectors

    for blob in container_client.list_blobs():
        blob_name = blob.name
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data = blob_client.download_blob().readall()

        metadata = yaml.safe_load(blob_data)
        summary_val = generate_summary(metadata)
        vectors = generate_sentence_vector(summary_val)

        yaml_vector_map[blob_name] = vectors  # Store the vectors with the blob name as key

        # print(f"Vectors for {blob_name}: {vectors} ")

    return yaml_vector_map


        
  
        
