# Import necessary classes
from langchain.document_loaders import GoogleApiClient
from langchain.document_loaders import GoogleApiYoutubeLoader
from langchain.indexes import VectorstoreIndexCreator
from pathlib import Path

# Define the path to the Google API service account file
service_account_path = Path("path_to_your_sec_file.json")

# Instantiate the Google API client
google_api_client = GoogleApiClient(service_account_path=service_account_path)

# Define the name of the YouTube channel to load videos from
channel_name = "CodeAesthetic"

# Instantiate the YouTube loader with the Google API client and the channel name
loader = GoogleApiYoutubeLoader(google_api_client=google_api_client, channel_name=channel_name)

# Load the documents (video transcripts and metadata) from the YouTube channel
documents = loader.load()

# Initialize a VectorstoreIndexCreator instance
index_creator = VectorstoreIndexCreator()

# Create an index from the loaded documents
index = index_creator.from_loaders([loader])

# Define the query string
query = "What is the channel about"

# Query the created index with the query string
response = index.query(query)

# Print the response
print(response)
