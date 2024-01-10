from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv, dotenv_values
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential
from scripts.prepdocs import CogIndexer

import os
import openai
import time
import glob
import io

open_ai_token_cache = {}
CACHE_KEY_TOKEN_CRED = 'openai_token_cred'
CACHE_KEY_CREATED_TIME = 'created_time'
CACHE_KEY_TOKEN_TYPE = 'token_type'

load_dotenv()

def process_file(filename, file_content=None, io_data=None):
    args={
        "skipblobs"      :False,
        "category"       :"public_records",
        "index"          :"publications",
        "novectors"      :False,
        "localpdfparser" :False,
        "remove"         :False,
        "removeall"      :False,
        "search_creds"   : AzureKeyCredential(os.getenv('SEARCHKEY')),
        "storage_creds"  : os.getenv('STORAGEKEY'),
        "formrecognizer_creds":AzureKeyCredential(os.getenv('FORMRECOGNIZERKEY')) 
    }
    
    CI = CogIndexer(args=args)
    
    # Use the current user identity to connect to Azure services unless a key is explicitly set for any of them
    azd_credential = AzureDeveloperCliCredential() if os.getenv('TENANTID') is None else AzureDeveloperCliCredential(tenant_id=os.getenv('TENANTID'), process_timeout=60)
    default_creds = azd_credential if os.getenv('SEARCHKEY') is None or os.getenv('STORAGEKEY') is None else None
    
    use_vectors = not args["novectors"]

    if not args["skipblobs"]:
        storage_creds = args["storage_creds"]
    if not args["localpdfparser"]:
        # check if Azure Form Recognizer credentials are provided
        if os.getenv('FORMRECOGNIZERSERVICE') is None:
            print("Error: Azure Form Recognizer service is not provided. Please provide formrecognizerservice or use --localpdfparser for local pypdf parser.")
            exit(1)
        formrecognizer_creds = args["formrecognizer_creds"]

    if use_vectors:
        if os.getenv('OPENAIKEY') is None:
            openai.api_key = azd_credential.get_token("https://cognitiveservices.azure.com/.default").token
            openai.api_type = "azure_ad"
            open_ai_token_cache[CACHE_KEY_CREATED_TIME] = time.time()
            open_ai_token_cache[CACHE_KEY_TOKEN_CRED] = azd_credential
            open_ai_token_cache[CACHE_KEY_TOKEN_TYPE] = "azure_ad"
        else:
            openai.api_type = "azure"
            openai.api_key = os.getenv('OPENAIKEY')

        openai.api_base = f"https://{os.getenv('OPENAISERVICE')}.openai.azure.com"
        openai.api_version = "2022-12-01"

    if args["removeall"]:
        CI.remove_blobs(None)
        CI.remove_from_index(None)
    else:
        if not args["remove"]:
            CI.create_search_index()

        print("Processing files...")
        
        
        if args["remove"]:
            CI.remove_blobs(filename)
            CI.remove_from_index(filename)
        elif args["removeall"]:
            CI.remove_blobs(None)
            CI.remove_from_index(None)
        else:
            if not args["skipblobs"]:
                CI.upload_blobs(filename, file_contents=io_data)
            page_map = CI.get_document_text(filename=io_data)
            sections = CI.create_sections(os.path.basename(filename), page_map, use_vectors)
            CI.index_sections(os.path.basename(filename), sections)

def iterate_blob():
    # Blob storage connection string
    KEY = os.getenv("BLOB_ACCOUNT_KEY")
    
    # Create a BlobServiceClient object
    account_url=f"https://{os.getenv('blob_trigger_account')}.blob.core.windows.net"
    print(account_url)
    blob_service_client = BlobServiceClient(account_url=account_url, credential=KEY)
    
    # List out the files in the container
    container_name = "openaiindexer"
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        print(blob.name)
    
        # Read a file from the container
        file_name = blob.name
        blob_client = container_client.get_blob_client(file_name)
        file_content = blob_client.download_blob().readall()
        data = io.BytesIO(file_content)
        process_file(file_name, file_content=file_content, io_data=data)
        blob_client.delete_blob()
        


