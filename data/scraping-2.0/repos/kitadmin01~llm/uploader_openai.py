import openai
from openai import OpenAI
import pinecone

class EmbeddingUploader:
    '''
    This class uses OpenAI to create the embeedings that are uploaded to PineCone for
    later search.
    '''
    def __init__(self, pinecone_api_key, pinecone_env, index_name, openai_api_key):
        # Initialize Pinecone with the provided API key and environment.
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

        # Check if the specified index already exists in Pinecone; if so, delete it.
        existing_indexes = pinecone.list_indexes()
        if index_name in existing_indexes:
            pinecone.delete_index(index_name)

        # Create a new Pinecone index with the specified name and a fixed dimension size.
        pinecone.create_index(index_name, dimension=1536)  
        self.index = pinecone.Index(index_name)

        # Initialize the OpenAI client with the provided API key.
        openai.api_key = openai_api_key
        self.client = OpenAI()

    def read_file_segments(self, file_path):
        # Read the content of a file and split it into segments based on empty lines.
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        segments = []
        segment = ""
        for line in lines:
            if line.strip():
                segment += line.strip() + " "
            else:
                if segment:
                    segments.append(segment.strip())
                    segment = ""

        if segment:
            segments.append(segment.strip())

        return segments

    def generate_embedding(self, text, model="text-embedding-ada-002"):
        # Generate an embedding for the given text using a specified OpenAI model.
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def upload_embeddings(self, file_paths):
        # Process each file and upload its embeddings to the Pinecone index.
        vectors_to_upsert = {}
        for file_type, file_path in file_paths.items():
            segments = self.read_file_segments(file_path)
            for i, segment in enumerate(segments):
                embedding = self.generate_embedding(segment)
                vectors_to_upsert[f"{file_type}_{i}"] = {
                    "values": embedding,
                    "metadata": {"text": segment, "type": file_type}
                }

        self.batch_upsert(vectors_to_upsert)

    def batch_upsert(self, data):
        # Batch upsert operation to add vectors to the Pinecone index in batches.
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch_data = [(k, v['values'], v['metadata']) for k, v in list(data.items())[i:i + batch_size]]
            self.index.upsert(vectors=batch_data)

if __name__ == "__main__":
    # Main execution: Initialize the EmbeddingUploader and upload embeddings for specified files.
    pinecone_api_key = "your-pinecone-api-key"
    pinecone_env = "gcp-starter"
    pinecone_index = "kit"
    openai_api_key = 'your-openai-api-key'

    uploader = EmbeddingUploader(pinecone_api_key, pinecone_env, pinecone_index, openai_api_key)
    path = "./aikit/content/"
    file_paths = {
        "faq": path + "faq.txt",
        "tool_use": path + "tool_use.txt",
        "tool_concept": path + "tool_concept.txt"
    }

    uploader.upload_embeddings(file_paths)
