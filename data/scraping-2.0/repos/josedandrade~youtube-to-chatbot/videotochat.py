# Import necessary libraries
from langchain.document_loaders import YoutubeLoader
from langchain.indexes import VectorstoreIndexCreator

# Load video from YouTube using YoutubeLoader
# The URL points to a specific video and add_video_info parameter is set to False
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=O5xeyoRL95U&ab_channel=LexFridman", 
    add_video_info=False
)

# Initialize a VectorstoreIndexCreator instance
index_creator = VectorstoreIndexCreator()

# Create an index from the loaded video
index = index_creator.from_loaders([loader])

# Define the query string
query = "What did the president say about Ketanji Brown Jackson"

# Query the created index with the query string
response = index.query(query)

# Optionally, print the response
print(response)
