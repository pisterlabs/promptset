from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from azure.storage.blob import BlobServiceClient
import zipfile
from fastapi.responses import RedirectResponse
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import subprocess
from typing import List
from azure.keyvault.secrets import SecretClient

app = FastAPI()

# try:
#     # Get the connection string from an environment variable
#     connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
# except Exception as e:
#     print(f"Error accessing environment variable for Azure: {e}")

def check_managed_identity_endpoint():
    try:
        # Define the command to access the IMDS endpoint
        command = "curl 'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://storage.azure.com/' -H 'Metadata: true'"
        # Execute the command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Read the output and error (if any)
        result, error = process.communicate()

        if process.returncode == 0:
            # Parse the result if the call to the endpoint was successful
            token_response = json.loads(result)
            print(f"IMDS endpoint response: {token_response}")
            return token_response
        else:
            # Output the error if the call failed
            print(f"Error contacting IMDS endpoint: {error}")
            return None
    except Exception as e:
        print(f"Exception when trying to access IMDS endpoint: {e}")
        return None

# Call the function to check the managed identity endpoint
#token_response = check_managed_identity_endpoint()

try:
    # Create a blob client using the storage account's connection string
    #blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a BlobServiceClient object using the DefaultAzureCredential
    managed_identity_client_id = os.getenv("MANAGED_IDENTITY_CLIENT_ID")
    credential = DefaultAzureCredential(managed_identity_client_id=managed_identity_client_id)
    #credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url="https://rmiinbox.blob.core.windows.net", credential=credential)
    #blob_service_client = BlobServiceClient(account_url="rmibox.blob.core.windows.net", credential=DefaultAzureCredential())

    # Specify the container and blob name
    container_name = "vectordb"
    blob_name = "data.zip"

    # Get a blob client for downloading
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Download the vectordb and save it locally
    local_path = "./data.zip"
    with open(local_path, "wb") as f:
        data = blob_client.download_blob()
        data.readinto(f)

    file_size = os.path.getsize(local_path)
    print(f"Downloaded file size: {file_size} bytes")

except Exception as e:
    print(f"Error while handling blob: {e}")


# Unzip the downloaded data
with zipfile.ZipFile(local_path, 'r') as zip_ref:
    zip_ref.extractall("./data")
print("Unzipping completed.")

# List files in the data directory
data_dir = './data'
print(f"Contents of {data_dir}:")
for filename in os.listdir(data_dir):
    print(filename)

# Create vector store
#from dotenv import load_dotenv
#load_dotenv('/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/envs/azure_storage.env')
#embedding = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
try:
    key_vault_uri = "https://rmibox.vault.azure.net/"
    client = SecretClient(vault_url=key_vault_uri, credential=credential)
    secret_name = "openai-api"
    retrieved_secret = client.get_secret(secret_name)
    print(f"Retrieved secret: {retrieved_secret.value}")
    embedding = OpenAIEmbeddings(openai_api_key=retrieved_secret.value)
except Exception as e:
    print(f"Error retrieving secret: {e}")
persist_directory = "./data/data"
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
print("Vector store initialized.")

# create Q&A chain
num_docs = 6
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=retrieved_secret.value),
    retriever=vectordb.as_retriever(search_kwargs={'k': num_docs}),
    return_source_documents=True,
    verbose=False,
)

#chat_history = []

# Use a dictionary as a mock in-memory database
sessions = {}

# Pydantic models for request and response
class AskRequest(BaseModel):
    uid: str
    query: str

class Source(BaseModel):
    source: str
    page: int

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]

# Redirect the default URL to the OpenAPI docs
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")

# Endpoint for asking a question
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    uid = request.uid
    query = request.query

    if not query:
        raise HTTPException(status_code=400, detail="Query not provided")

    if not uid:
        raise HTTPException(status_code=400, detail="UID not provided")

    # Create a dictionary entry for the user if it doesn't exist
    if uid not in sessions:
        sessions[uid] = []

    # Get the chat history for the user
    chat_history = sessions[uid]

    # Ask the question and get response
    result = pdf_qa({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    #sources = [{"source": doc.metadata["source"], "page": doc.metadata["page"]+1} for doc in result["source_documents"]]
    sources = [Source(source=doc.metadata["source"], page=doc.metadata["page"]+1) for doc in result["source_documents"]]

    # Update the chat history
    chat_history.append((query, answer))
    sessions[uid] = chat_history

    print(f"Response ready to be sent. Sources: {sources}")
    return AskResponse(answer=answer, sources=sources)

# Asynchronous entry point for FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)


#%%

# from fastapi import Request
# import json

# # Create an AskRequest object
# request = AskRequest(uid="55", query="Explain downstream oil emissions")

# # Call the endpoint function directly
# response = ask(request)

# print(response.answer)
# print(response.sources)
# print(len(response.sources))

#%%
