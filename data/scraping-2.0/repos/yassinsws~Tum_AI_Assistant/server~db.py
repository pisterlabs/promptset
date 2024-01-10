import json
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes._generated.models import CorsOptions, ScoringProfile, VectorSearch, \
    VectorSearchProfile, HnswAlgorithmConfiguration
from azure.search.documents.indexes.models import SimpleField, SearchableField, SearchFieldDataType, SearchIndex, \
    SearchField
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import openai
from secrets import token_urlsafe


class VectorDB:
    def __init__(self, api_key=None, api_endpoint=None, index=None, search_service="docu-search"):
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.getenv('AZURE_SEARCH_API_KEY')
        self.search_service = search_service
        self.index_name = index
        if not self.index_name:
            self.index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')

        self.url = f"https://{self.search_service}.search.windows.net/indexes/{self.index_name}/docs/search?api-version=2023-10-01-preview"

        self.fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="needs_approval", type=SearchFieldDataType.String),
            SearchableField(name="contact_person", type=SearchFieldDataType.String),
            SearchableField(name="source_url", type=SearchFieldDataType.String),
            # SearchableField(name="summary", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(name="descriptionVector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True,
                        vector_search_dimensions=1536,
                        vector_search_profile_name="vector-profile-1701507968183"
                        ),
            SearchField(name="document_title", type=SearchFieldDataType.String, searchable=True),
        ]
        self.api_endpoint = api_endpoint
        if not self.api_endpoint:
            self.api_endpoint = os.getenv('AZURE_SEARCH_API_ENDPOINT')
        self.credentials = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(self.api_endpoint, self.credentials)
        self.search_client = SearchClient(self.api_endpoint, self.index_name, self.credentials)

        self.cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
        self.scoring_profiles: List[ScoringProfile] = []
        self.vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(name="vector-profile-1701507968183",
                                    algorithm_configuration_name="vector-config-1701504391465")],
            algorithms=[HnswAlgorithmConfiguration(name="vector-config-1701504391465")],
        )
        self.search_index = SearchIndex(name=self.index_name, fields=self.fields, vector_search=self.vector_search,
                                        scoring_profiles=self.scoring_profiles, cors_options=self.cors_options)

    def get_embeddings(self, text: str):
        # There are a few ways to get embeddings. This is just one example.
        open_ai_endpoint = "https://teamiris.openai.azure.com/"
        open_ai_key = os.getenv("AZURE_OPENAI_API_KEY")
        client = openai.AzureOpenAI(
            azure_endpoint=open_ai_endpoint,
            api_key=open_ai_key,
            api_version="2023-09-01-preview",
        )
        embedding = client.embeddings.create(input=[text], model="text-embedding-ada")
        return embedding.data[0].embedding

    async def upload_new_pdf(self, path_to_pdf, old_name, contact_person, url):
        def extract_text_from_pdf(file_path):
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ''
                    for page in range(len(reader.pages)):
                        text += reader.pages[page].extract_text()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1024,
                        chunk_overlap=123,
                        length_function=len,
                        is_separator_regex=False,
                    )
                    texts = text_splitter.split_text(text)
                    return texts
            except Exception as e:
                print(f"An error occurred: {e}")
                return None

        if old_name:
            result = self.search_client.search(search_text=old_name)
            df = pd.DataFrame(result)
            df = df[df["document_title"] == old_name]["id"].to_numpy()
            print(df)
            if len(df) > 0:
                self.search_client.delete_documents(documents=[{"id": i} for i in df])
                print(self.search_client.get_document_count())

        pdf_text = extract_text_from_pdf(path_to_pdf)
        assert pdf_text, "Error trying to upload new pdf, could not extract text from pdf!"

        for i, text in enumerate(pdf_text):
            payload = {
                "@search.action": "upload",
                "document_title": old_name,
                "id": f'{old_name}{i}',
                "contact_person": contact_person,
                "source_url": url,
                # "summary":"Modulehandbook_MBA Executive Master of Business Administration in Business & IT",
                "descriptionVector": self.get_embeddings(text),
                "content": text
            }
            response = self.search_client.upload_documents(payload)
    async def query(self, query_text, keywords, return_k=50):
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key  # Replace with your actual API key
        }
        payload = {
            "select": "content, document_title, source_url, contact_person",
            "search": keywords,
            "top": 5,
            "vectorQueries": [
                {
                    "vector": self.get_embeddings(keywords),
                    "k": return_k,
                    "fields": "descriptionVector",
                    "kind": "vector",
                    "exhaustive": True
                }
            ],
        }
        response = requests.post(self.url, headers=headers, data=json.dumps(payload))

        # Check the response
        assert response.status_code == 200, "Error {} {}".format(response.status_code, response.text)
        print('DATABASE RESPONSE!')
        print(response.json())
        return response.json()['value']


if __name__ == "__main__":
    db = VectorDB()
    print(db.query(query_text="Master Thesis"))
