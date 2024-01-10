from llama_index import VectorStoreIndex, SimpleDirectoryReader
import os
import openai

#gets api key from environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

#loads the data from the data folder (These can be PDFs, Txt files, or any other file type that can be read by llama_index)
documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)


#saves the index to the storage folder
index.storage_context.persist()


print("Indexing Complete!")