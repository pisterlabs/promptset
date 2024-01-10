"""
This will be the file handling the interfacing between the dev user agent repo and the app-template.
If a file is uploaded to the agent, it will be stored in the fodler agent/tmp/{session_id}/
"""

from dotenv import load_dotenv
import os
import pinecone
import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load env variables from .env file
load_dotenv()

# Get env variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Auth to pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

embeddings = OpenAIEmbeddings()

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(PINECONE_INDEX_NAME)

vectorstore = Pinecone(
    index, embeddings.embed_query, text_field
)

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

### INTERFACE ###

def ask(messages, session_id):

    if len(messages) == 0:
        response = "No message received"
    else:
        # YOUR CODE HERE
        last_message = messages[-1]
        last_message_content = last_message["content"]
        # Log the message
        print(f"Received message: {last_message_content}")
        response = qa.run(last_message_content)
    # You need to return a string, that will be displayed to the user as a message
    return response #str

# Handle a single file upload
def handle_single_file_upload(session_id, sanitized_filename):
    # YOUR CODE HERE
    # The file is at agent/tmp/{session_id}/{sanitized_filename}
    # You can delete the file with os.remove(path) to free up memory
    pass # no need to return anything

# Handle a single file download request
def handle_single_file_download(session_id):
    # YOUR CODE HERE
    # You need to return the relative path to the file to download (e.g. 'tmp/{session_id}/{filename}')
    # The file will be downloaded in your end user browser
    path_to_file = "path/to/file"
    return path_to_file # type str