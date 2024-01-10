import os
import json
import openai
import requests
import base64
from tqdm import tqdm
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import time

load_dotenv()

# create indexer class
class Indexer:
    ### Create Azure Search Vector-based Index
    # Setup the Payloads header

    def __init__(self, blob_conatiner_name: str="pci-dss", index_name: str="pci_dss_index", isVector: bool=False):
        openai.api_type = "azure"
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = "2023-03-15-preview"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.index_name = index_name
        self.isVector = isVector
        self.blob_conatiner_name = blob_conatiner_name
        self.headers = {'Content-Type': 'application/json','api-key': os.environ['AZURE_SEARCH_KEY']}
        self.params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}
        self.url = os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + self.index_name + "/docs/index"
        self.blob_service_client = BlobServiceClient.from_connection_string(os.getenv("CONNECTION_STRING"))
        self.container_client = self.blob_service_client.get_container_client(blob_conatiner_name)
        self.blob_list = self.container_client.list_blobs()

    def run(self):
        self.create_index_azure_search(self.index_name, self.isVector)
        self.upload_index_azure_search(self.index_name, self.isVector)

    def text_to_base64(self, text):
        # Convert text to bytes using UTF-8 encoding
        bytes_data = text.encode('utf-8')

        # Perform Base64 encoding
        base64_encoded = base64.b64encode(bytes_data)

        # Convert the result back to a UTF-8 string representation
        base64_text = base64_encoded.decode('utf-8')

        return base64_text

    def create_index_azure_search(self, index_name: str="demo_index_0", isVector: bool=False):
        if isVector:
            index_payload = {
                "name": index_name,
                "fields": [
                    {"name": "id", "type": "Edm.String", "key": "true", "filterable": "true" },
                    {"name": "title","type": "Edm.String","searchable": "true","retrievable": "true"},
                    {"name": "content","type": "Edm.String","searchable": "true","retrievable": "true"},
                    {"name": "contentVector","type": "Collection(Edm.Single)","searchable": "true","retrievable": "true","dimensions": 1536,"vectorSearchConfiguration": "vectorConfig"},
                    {"name": "name", "type": "Edm.String", "searchable": "true", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},
                    {"name": "location", "type": "Edm.String", "searchable": "false", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},     
                    {"name": "page_num","type": "Edm.Int32","searchable": "false","retrievable": "true"},
                ],
                "vectorSearch": {
                    "algorithmConfigurations": [
                        {
                            "name": "vectorConfig",
                            "kind": "hnsw"
                        }
                    ]
                },
                "semantic": {
                    "configurations": [
                        {
                            "name": "my-semantic-config",
                            "prioritizedFields": {
                                "titleField": {
                                    "fieldName": "title"
                                },
                                "prioritizedContentFields": [
                                    {
                                        "fieldName": "content"
                                    }
                                ],
                                "prioritizedKeywordsFields": []
                            }
                        }
                    ]
                }
            }
        else:
            index_payload = {
                "name": index_name,
                "fields": [
                    {"name": "id", "type": "Edm.String", "key": "true", "filterable": "true" },
                    {"name": "title","type": "Edm.String","searchable": "true","retrievable": "true"},
                    {"name": "content","type": "Edm.String","searchable": "true","retrievable": "true"},
                    {"name": "name", "type": "Edm.String", "searchable": "true", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},
                    {"name": "location", "type": "Edm.String", "searchable": "false", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},
                    {"name": "page_num","type": "Edm.Int32","searchable": "false","retrievable": "true"},
                ],
                "semantic": {
                    "configurations": [
                        {
                            "name": "my-semantic-config",
                            "prioritizedFields": {
                                "titleField": {
                                    "fieldName": "title"
                                },
                                "prioritizedContentFields": [
                                    {
                                        "fieldName": "content"
                                    }
                                ],
                                "prioritizedKeywordsFields": []
                            }
                        }
                    ]
                }
            }

        r = requests.put(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index_name, data=json.dumps(index_payload), headers=self.headers, params=self.params)
        print(r.status_code)
        print(r.ok)    

    def upload_index_azure_search(self, index_name: str="demo_index_0", isVector: bool=False):
        print("Upload data to index")
        if isVector:
            blob_service_client = BlobServiceClient.from_connection_string(os.getenv("CONNECTION_STRING"))
            # get a list of blobs using blob_service_client
            container_client = blob_service_client.get_container_client(self.blob_conatiner_name)
            blob_list = container_client.list_blobs()
            # read data in the blob
            for blob in blob_list:
                blob_client = blob_service_client.get_blob_client(container=self.blob_conatiner_name, blob=blob.name)
                # print(blob_client.url)
                # get file name for the blob.url to use it as document_name
                document_name = blob.name.split("/")[-1].split(".")[0]        
                page_num = int(document_name.split("-")[-1])
                blob_data = blob_client.download_blob().readall()
                blob_data = blob_data.decode("utf-8", "ignore")
                title = blob_data.split("\n")[0].replace("\n", "").replace("\r", "")
                try:
                    upload_payload = {
                        "value": [
                            {
                                "id": self.text_to_base64(document_name),
                                "title": f"{title[:45]}",
                                "content": blob_data,
                                "contentVector": openai.Embedding.create(input=[blob_data], engine="text-embedding-ada-002")["data"][0]["embedding"],
                                "name": document_name,
                                "location": blob_client.url,
                                "page_num": page_num,
                                "@search.action": "upload"
                            },
                        ]
                    }
                    
                    r = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index_name + "/docs/index", data=json.dumps(upload_payload), headers=self.headers, params=self.params)
                    print(r.status_code)
                    print(r.text)
                except Exception as e:
                    print("Exception:",e)
                    # print(content)
                    continue
                # time.sleep(1)
        else:
            blob_service_client = BlobServiceClient.from_connection_string(os.getenv("CONNECTION_STRING"))
            # get a list of blobs using blob_service_client
            container_client = blob_service_client.get_container_client(self.blob_conatiner_name)
            blob_list = container_client.list_blobs()
            # read data in the blob
            for blob in blob_list:
                blob_client = blob_service_client.get_blob_client(container=self.blob_conatiner_name, blob=blob.name)
                # get file name for the blob.url to use it as document_name
                document_name = blob.name.split("/")[-1].split(".")[0]        
                page_num = int(document_name.split("-")[-1])
                blob_data = blob_client.download_blob().readall()
                blob_data = blob_data.decode("utf-8", "ignore")
                title = blob_data.split("\n")[0].replace("\n", "").replace("\r", "")
                try:
                    upload_payload = {
                        "value": [
                            {
                                "id": self.text_to_base64(document_name),
                                "title": f"{title[:45]}",
                                "content": blob_data,
                                "name": document_name,
                                "location": blob_client.url,
                                "page_num": page_num,
                                "@search.action": "upload"
                            },
                        ]
                    }
                    
                    r = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index_name + "/docs/index", data=json.dumps(upload_payload), headers=self.headers, params=self.params)
                    if r.status_code != 200:
                        print(r.status_code)
                        print(r.text)
                except Exception as e:
                    print("Exception:",e)
                    continue
                # time.sleep(1)
            
