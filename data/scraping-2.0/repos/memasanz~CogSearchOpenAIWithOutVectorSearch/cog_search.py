import requests
import json

import json
import numpy as np
import os
import pandas as pd
import openai

from collections import OrderedDict
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
from dotenv import dotenv_values

class CogSearchHelper:
    def __init__(self, index):
        self.service_name = os.getenv('COG_SEARCH_RESOURCE')
        self.usgov = os.getenv('USGOV')
        if self.usgov == 'True':
            self.endpoint = "https://{}.search.azure.us/".format(self.service_name)
        else:
            self.endpoint = "https://{}.search.windows.net/".format(self.service_name)
        print(self.endpoint)
        self.search_key = os.getenv('COG_SEARCH_KEY')
        self.storage_connectionstring = os.getenv('STORAGE_CONNECTION_STRING')
        self.storage_container = os.getenv('STORAGE_CONTAINER')
        self.cognitive_service_key = os.getenv('COG_SERVICE_KEY')
        
        if index == None:
            self.index = os.getenv('COG_SEARCH_INDEX')
        else:
            self.index = index
    
    def get_the_token_count(self, documents):
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        total_token_count = 0
        try:
            token_count = len(tokenizer.encode(documents))
        except:
            print('failed to get token count')
            token_count = -1
            pass

        return token_count
    
    def search_single_docs(df, user_query, TEXT_SEARCH_QUERY_EMBEDDING_ENGINE, top_n=3):
        embedding = get_embedding(
            user_query,
            engine=TEXT_SEARCH_QUERY_EMBEDDING_ENGINE
        )
        df["similarities"] = df.curie_search.apply(lambda x: cosine_similarity(x, embedding))

        res = (
            df.sort_values("similarities", ascending=False)
            .reset_index(drop=True)
            .head(top_n)
        )
        return res

    def search(self, question):
        response = openai.Embedding.create(input=question,engine="text-embedding-ada-002")
        q_embeddings = response['data'][0]['embedding']
        
        if len(question) > 0:
            endpoint = "https://{}.search.windows.net/".format(self.service_name)
            url = '{0}indexes/{1}/docs/search?api-version=2021-04-30-Preview'.format(endpoint, self.index)

            print(url)

            payload = json.dumps({
            "search": question,
            "count": True,
            })
            headers = {
            'api-key': '{0}'.format(self.search_key),
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)
            obj = response.json()
            relevant_data = []
            lst_embeddings_text = []
            lst_embeddings = []
            lst_file_name = []
            count = 0
            for x in obj['value']:
                if x['@search.score'] > 0.5:
                    count += 1
                    relevant_data.append(x['content'])
                    embeddings = x['embeddings']
                    embeddings_text = x['embeddings_text']
                    file_name = x['metadata_storage_name']

                    curie_search = []
                    for x in embeddings:
                        a = np.fromstring(x[1:-1], dtype=float, sep=',')
                        curie_search.append(a)
                    curie_list = list(curie_search)

                    for i in range(len(embeddings)):
                        lst_embeddings_text.append(embeddings_text[i])
                        lst_embeddings.append(np.fromstring(embeddings[i][1:-1], dtype=float, sep=','))
                        lst_file_name.append(file_name)
                

            tuples_list = []
            metadata_list = []
            tokencount = 0
            for i in range(len(lst_embeddings_text)):
                tuples_list.append((lst_embeddings_text[i], lst_embeddings[i]))
                metadata_list.append(dict(source=lst_file_name[i]))


            return relevant_data, count, lst_file_name, tuples_list, lst_embeddings_text, metadata_list
    
    #COG_SEARCH_RESOURCE, COG_SEARCH_INDEX, COG_SEARCH_KEY, STORAGE_CONNECTION_STRING, STORAGE_CONTAINER
    def create_datasource(self):

        url = '{0}/datasources/{1}-datasource?api-version=2020-06-30'.format(self.endpoint, self.index)

        payload = json.dumps({
                    "description": "Demo files to demonstrate cognitive search capabilities.",
                    "type": "azureblob",
                    "credentials": {
                        "connectionString": self.storage_connectionstring
                    },
                    "container": {
                        "name": self.storage_container
                    }
                    })
        headers = {
        'api-key': self.search_key,
        'Content-Type': 'application/json'
                }

        response = requests.request("PUT", url, headers=headers, data=payload)

        if response.status_code == 201 or response.status_code == 204:
            return response, True
        else:
            print(response.json())
            return response, False
    
    def create_skillset(self, cognitive_service_key, embeddingFunctionAppUriAndKey):
        url = '{0}/skillsets/{1}-skillset?api-version=2021-04-30-Preview'.format(self.endpoint, self.index)
        print(url)
        payload = json.dumps({
        "@odata.context": "{}/$metadata#skillsets/$entity".format(self.endpoint),
        "@odata.etag": "\"0x8DB2B4BF82370CF\"",
        "name": "{0}-skillset".format(self.index),
        "description": "Skillset created from the portal. skillsetName: index-skillset; contentField: merged_content; enrichmentGranularity: document; knowledgeStoreStorageAccount: ;",
        "skills": [
            {
            "@odata.type": "#Microsoft.Skills.Text.V3.EntityRecognitionSkill",
            "name": "#1",
            "description": None,
            "context": "/document/merged_content",
            "categories": [
                "Organization",
                "URL",
                "DateTime",
                "Skill",
                "Address",
                "Location",
                "Product",
                "IPAddress",
                "Event",
                "Person",
                "Quantity",
                "PersonType",
                "PhoneNumber",
                "Email"
            ],
            "defaultLanguageCode": "en",
            "minimumPrecision": None,
            "modelVersion": None,
            "inputs": [
                {
                "name": "text",
                "source": "/document/merged_content"
                },
                {
                "name": "languageCode",
                "source": "/document/language"
                }
            ],
            "outputs": [
                {
                "name": "persons",
                "targetName": "people"
                },
                {
                "name": "organizations",
                "targetName": "organizations"
                },
                {
                "name": "locations",
                "targetName": "locations"
                }
            ]
            },
            {
            "@odata.type": "#Microsoft.Skills.Text.KeyPhraseExtractionSkill",
            "name": "#2",
            "description": None,
            "context": "/document/merged_content",
            "defaultLanguageCode": "en",
            "maxKeyPhraseCount": None,
            "modelVersion": None,
            "inputs": [
                {
                "name": "text",
                "source": "/document/merged_content"
                },
                {
                "name": "languageCode",
                "source": "/document/language"
                }
            ],
            "outputs": [
                {
                "name": "keyPhrases",
                "targetName": "keyphrases"
                }
            ]
            },
            {
            "@odata.type": "#Microsoft.Skills.Text.LanguageDetectionSkill",
            "name": "#3",
            "description": None,
            "context": "/document",
            "defaultCountryHint": None,
            "modelVersion": None,
            "inputs": [
                {
                "name": "text",
                "source": "/document/merged_content"
                }
            ],
            "outputs": [
                {
                "name": "languageCode",
                "targetName": "language"
                }
            ]
            },
            {
            "@odata.type": "#Microsoft.Skills.Text.TranslationSkill",
            "name": "#4",
            "description": None,
            "context": "/document/merged_content",
            "defaultFromLanguageCode": None,
            "defaultToLanguageCode": "en",
            "suggestedFrom": "en",
            "inputs": [
                {
                "name": "text",
                "source": "/document/merged_content"
                }
            ],
            "outputs": [
                {
                "name": "translatedText",
                "targetName": "translated_text"
                }
            ]
            },
            {
            "@odata.type": "#Microsoft.Skills.Text.PIIDetectionSkill",
            "name": "#5",
            "description": None,
            "context": "/document/merged_content",
            "defaultLanguageCode": "en",
            "minimumPrecision": 0.5,
            "maskingMode": "replace",
            "maskingCharacter": "*",
            "modelVersion": None,
            "piiCategories": [],
            "domain": "none",
            "inputs": [
                {
                "name": "text",
                "source": "/document/merged_content"
                },
                {
                "name": "languageCode",
                "source": "/document/language"
                }
            ],
            "outputs": [
                {
                "name": "piiEntities",
                "targetName": "pii_entities"
                },
                {
                "name": "maskedText",
                "targetName": "masked_text"
                }
            ]
            },
            {
            "@odata.type": "#Microsoft.Skills.Text.MergeSkill",
            "name": "#6",
            "description": None,
            "context": "/document",
            "insertPreTag": " ",
            "insertPostTag": " ",
            "inputs": [
                {
                "name": "text",
                "source": "/document/content"
                },
                {
                "name": "itemsToInsert",
                "source": "/document/normalized_images/*/text"
                },
                {
                "name": "offsets",
                "source": "/document/normalized_images/*/contentOffset"
                }
            ],
            "outputs": [
                {
                "name": "mergedText",
                "targetName": "merged_content"
                }
            ]
            },
            {
            "@odata.type": "#Microsoft.Skills.Vision.OcrSkill",
            "name": "#7",
            "description": None,
            "context": "/document/normalized_images/*",
            "textExtractionAlgorithm": None,
            "lineEnding": "Space",
            "defaultLanguageCode": "en",
            "detectOrientation": True,
            "inputs": [
                {
                "name": "image",
                "source": "/document/normalized_images/*"
                }
            ],
            "outputs": [
                {
                "name": "text",
                "targetName": "text"
                },
                {
                "name": "layoutText",
                "targetName": "layoutText"
                }
            ]
            },
            {
            "@odata.type": "#Microsoft.Skills.Vision.ImageAnalysisSkill",
            "name": "#8",
            "description": None,
            "context": "/document/normalized_images/*",
            "defaultLanguageCode": "en",
            "visualFeatures": [
                "tags",
                "description"
            ],
            "details": [],
            "inputs": [
                {
                "name": "image",
                "source": "/document/normalized_images/*"
                }
            ],
            "outputs": [
                {
                "name": "tags",
                "targetName": "imageTags"
                },
                {
                "name": "description",
                "targetName": "imageCaption"
                }
            ]
            }
            ,
            {
    "@odata.type": "#Microsoft.Skills.Custom.WebApiSkill",
    "uri": embeddingFunctionAppUriAndKey,
    "httpMethod": "POST",
    "timeout": "PT230S",
    "batchSize": 1,
    "degreeOfParallelism": 1,
    "name": "Embeddings",
    "description": "",
    "context": "/document",
    "inputs": [
            {
            "name": "text",
            "source": "/document/merged_content"
            }
    ],
    "outputs": [
            {
                "name": "embeddings",
                "targetName": "embeddings"
            },
                        {
                "name": "embeddings_text",
                "targetName": "embeddings_text"
            }
    ]
    }

        ],
        "cognitiveServices": {
            "@odata.type": "#Microsoft.Azure.Search.CognitiveServicesByKey",
            "description": "SuperCool",
            "key": "{0}".format(cognitive_service_key)
        },
        "knowledgeStore": None,
        "encryptionKey": None
        })
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': '{0}'.format(self.search_key)
        }

        
        response = requests.request("PUT", url, headers=headers, data=payload)

        print(response.text)

        if response.status_code == 201 or response.status_code == 204:
            return response, True
        else:

            return response, False
    
    def update_index_semantic(self):

        url = '{0}/indexes/{1}/?api-version=2021-04-30-Preview'.format(self.endpoint, self.index)
        print(url)

        payload = json.dumps({
        "name": self.index,
        "defaultScoringProfile": "",
        "fields": [
            {
            "name": "content",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "metadata_storage_content_type",
            "type": "Edm.String",
            "searchable": False,
            "filterable": False,
            "retrievable": False,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "synonymMaps": []
            },
            {
            "name": "metadata_storage_size",
            "type": "Edm.Int64",
            "searchable": False,
            "filterable": False,
            "retrievable": False,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "synonymMaps": []
            },
            {
            "name": "metadata_storage_last_modified",
            "type": "Edm.DateTimeOffset",
            "searchable": False,
            "filterable": False,
            "retrievable": False,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "synonymMaps": []
            },
            {
            "name": "metadata_storage_content_md5",
            "type": "Edm.String",
            "searchable": False,
            "filterable": False,
            "retrievable": False,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "synonymMaps": []
            },
            {
            "name": "metadata_storage_name",
            "type": "Edm.String",
            "searchable": False,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "synonymMaps": []
            },
            {
            "name": "metadata_storage_path",
            "type": "Edm.String",
            "searchable": False,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": True,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "synonymMaps": []
            },
            {
            "name": "metadata_storage_file_extension",
            "type": "Edm.String",
            "searchable": False,
            "filterable": False,
            "retrievable": False,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None, 
            "synonymMaps": []
            },
            {
            "name": "metadata_content_type",
            "type": "Edm.String",
            "searchable": False,
            "filterable": False,
            "retrievable": False,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "synonymMaps": []
            },
            {
            "name": "metadata_language",
            "type": "Edm.String",
            "searchable": False,
            "filterable": False,
            "retrievable": False,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "synonymMaps": []
            },
            {
            "name": "metadata_creation_date",
            "type": "Edm.DateTimeOffset",
            "searchable": False,
            "filterable": False,
            "retrievable": False,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "synonymMaps": []
            },
            {
            "name": "people",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "organizations",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "locations",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "keyphrases",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "language",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "translated_text",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "en.lucene",
            "synonymMaps": []
            },
            {
            "name": "embeddings_text",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "embeddings",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "pii_entities",
            "type": "Collection(Edm.ComplexType)",
            "fields": [
                {
                "name": "text",
                "type": "Edm.String",
                "searchable": True,
                "filterable": False,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": "standard.lucene",
                "synonymMaps": []
                },
                {
                "name": "type",
                "type": "Edm.String",
                "searchable": True,
                "filterable": False,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": "standard.lucene",
                "synonymMaps": []
                },
                {
                "name": "subtype",
                "type": "Edm.String",
                "searchable": True,
                "filterable": False,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": "standard.lucene",
                "synonymMaps": []
                },
                {
                "name": "offset",
                "type": "Edm.Int32",
                "searchable": False,
                "filterable": False,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": None,
                "synonymMaps": []
                },
                {
                "name": "length",
                "type": "Edm.Int32",
                "searchable": False,
                "filterable": False,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": None,
                "synonymMaps": []
                },
                {
                "name": "score",
                "type": "Edm.Double",
                "searchable": False,
                "filterable": False,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": None,
                "synonymMaps": []
                }
            ]
            },
            {
            "name": "masked_text",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "merged_content",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "text",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "layoutText",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "imageTags",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            },
            {
            "name": "imageCaption",
            "type": "Collection(Edm.String)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": "standard.lucene",
            "synonymMaps": []
            }
        ],
        "scoringProfiles": [],
        "corsOptions": None,
        "suggesters": [],
        "analyzers": [],
        "tokenizers": [],
        "tokenFilters": [],
        "charFilters": [],
        "encryptionKey": None,
        "similarity": {
            "@odata.type": "#Microsoft.Azure.Search.BM25Similarity",
            "k1": None,
            "b": None
        }
        })
        headers = {
        'api-key': self.search_key,
        'Content-Type': 'application/json'
        }

        response = requests.request("PUT", url, headers=headers, data=payload)

        if response.status_code == 201 or response.status_code == 204:
            return response, True
        else:
            # print('************************')
            # print(response.status_code)
            # print(response.text)
            return response, False

    def create_indexer(self):

        url = '{0}/indexers/{1}-indexer/?api-version=2021-04-30-Preview'.format(self.endpoint, self.index)
        print(url)

        payload = json.dumps({
        "name": "{0}-indexer".format(self.index),
        "description": "",
        "dataSourceName": "{0}-datasource".format(self.index),
        "skillsetName": "{0}-skillset".format(self.index),
        "targetIndexName": "{0}".format(self.index),
        "disabled": None,
        "schedule": None,
        "parameters": {
            "batchSize": None,
            "maxFailedItems": 0,
            "maxFailedItemsPerBatch": 0,
            "base64EncodeKeys": None,
            "configuration": {
            "dataToExtract": "contentAndMetadata",
            "parsingMode": "default",
            "imageAction": "generateNormalizedImages"
            }
        },
        "fieldMappings": [
            {
            "sourceFieldName": "metadata_storage_path",
            "targetFieldName": "metadata_storage_path",
            "mappingFunction": {
                "name": "base64Encode",
                "parameters": None
            }
            }
        ],
        "outputFieldMappings": [
            {
            "sourceFieldName": "/document/merged_content/people",
            "targetFieldName": "people"
            },
            {
            "sourceFieldName": "/document/merged_content/organizations",
            "targetFieldName": "organizations"
            },
            {
            "sourceFieldName": "/document/merged_content/locations",
            "targetFieldName": "locations"
            },
            {
            "sourceFieldName": "/document/merged_content/keyphrases",
            "targetFieldName": "keyphrases"
            },
            {
            "sourceFieldName": "/document/language",
            "targetFieldName": "language"
            },
            {
            "sourceFieldName": "/document/merged_content/translated_text",
            "targetFieldName": "translated_text"
            },
            {
            "sourceFieldName": "/document/merged_content/pii_entities",
            "targetFieldName": "pii_entities"
            },
            {
            "sourceFieldName": "/document/merged_content/masked_text",
            "targetFieldName": "masked_text"
            },
            {
            "sourceFieldName": "/document/merged_content",
            "targetFieldName": "merged_content"
            },
            {
            "sourceFieldName": "/document/normalized_images/*/text",
            "targetFieldName": "text"
            },
            {
            "sourceFieldName": "/document/normalized_images/*/layoutText",
            "targetFieldName": "layoutText"
            },
            {
            "sourceFieldName": "/document/normalized_images/*/imageTags/*/name",
            "targetFieldName": "imageTags"
            },
            {
            "sourceFieldName": "/document/normalized_images/*/imageCaption",
            "targetFieldName": "imageCaption"
            },
            {
            "sourceFieldName": "/document/embeddings",
            "targetFieldName": "embeddings"
            },
            {
            "sourceFieldName": "/document/embeddings_text",
            "targetFieldName": "embeddings_text"
            }
            
        ],
        "cache": None,
        "encryptionKey": None
        })
        headers = {
        'Content-Type': 'application/json',
        'api-key': '{0}'.format(self.search_key)
        }


        response = requests.request("PUT", url, headers=headers, data=payload)


        if response.status_code == 201 or response.status_code == 204:
            print('good')
            return response, True
        else:
            print(response.status_code)
            return response, False
    
class OpenAIHelper:
    def __init__(self, index):

        config = dotenv_values(".env")
        # Set the ENV variables that Langchain needs to connect to Azure OpenAI

        os.environ['AZURE_OPENAI_API_VERSION'] = "2023-05-15"
        os.environ["OPENAI_API_BASE"] = os.environ['AZURE_OPENAI_ENDPOINT']
        os.environ["OPENAI_API_KEY"] = os.environ['AZURE_OPENAI_KEY']
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        os.environ["OPENAI_API_TYPE"] = "azure"
        COMBINE_PROMPT_TEMPLATE = """
            These are examples of how you must provide the answer:

            --> Beginning of examples

            =========
            QUESTION: Which state/country's law governs the interpretation of the contract?
            =========
            Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
            Source: SuperCool.docx

            Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
            Source: https://yyyy.com/article2.html?s=lkhljkhljk&category=c&sort=asc

            Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
            Source: https://yyyy.com/article3.csv?s=kjsdhfd&category=c&sort=asc&page=2

            Content: The terms of this Agreement shall be subject to the laws of Manchester, England, and any disputes arising from or relating to this Agreement shall be exclusively resolved by the courts of that state, except where either party may seek an injunction or other legal remedy to safeguard their Intellectual Property Rights.
            Source: https://ppp.com/article4.pdf?s=lkhljkhljk&category=c&sort=asc
            =========
            FINAL ANSWER IN English: This Agreement is governed by English law, specifically the laws of Manchester, England<sup><a href="https://xxx.com/article1.pdf?s=casdfg&category=ab&sort=asc&page=1" target="_blank">[1]</a></sup><sup><a href="https://ppp.com/article4.pdf?s=lkhljkhljk&category=c&sort=asc" target="_blank">[2]</a></sup>. \n Anything else I can help you with?.

            =========
            QUESTION: What did the president say about Michael Jackson?
            =========
            Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny..
            Source: https://fff.com/article23.pdf?s=wreter&category=ab&sort=asc&page=1

            Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
            Source: https://jjj.com/article56.pdf?s=sdflsdfsd&category=z&sort=desc&page=3

            Content: And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
            Source: https://vvv.com/article145.pdf?s=sfsdfsdfs&category=z&sort=desc&page=3

            Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
            Source: https://uuu.com/article15.pdf?s=lkhljkhljk&category=c&sort=asc
            =========
            FINAL ANSWER IN English: The president did not mention Michael Jackson.

            <-- End of examples

            # Instructions:
            - Given the following extracted parts from one or multiple documents, and a question, create a final answer with references. 
        
            - **Answer the question from information provided in the context, DO NOT use your prior knowledge.
            - Never provide an answer without references.
            - If the question is one word, rephrase it to: "Tell me about a " and then the question
            - If you don't know the answer, respond with "I don't know the answer to that question. Please try rephrasing your question."
            - Respond in {language}.

            =========
            QUESTION: {question}
            =========
            {summaries}
            =========
            FINAL ANSWER IN {language}:"""

        self.COMBINE_PROMPT = PromptTemplate(template=COMBINE_PROMPT_TEMPLATE, input_variables=["summaries", "question", "language"])

        #self.question_template = creds['QUESTION_TEMPLATE']
        if index == None:
            self.index = os.environ['COG_SEARCH_INDEX']
        else:
            self.index = index

    def get_the_token_count(self, documents):
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        total_token_count = 0
        try:
            token_count = len(tokenizer.encode(documents))
        except:
            print('failed to get token count')
            token_count = -1
            pass

        return token_count
    
    def get_Answer_from_load_qa_with_sources_chain(self, question):
        print('get answer from load qa with source')
        openai.api_type = "azure"
        openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
        openai.api_version = "2022-12-01"
        os.environ['OPENAI_API_KEY'] = os.environ['AZURE_OPENAI_KEY']
        openai.api_key = os.getenv("OPENAI_API_KEY")

        from openai.embeddings_utils import get_embedding, cosine_similarity

        question_embedding = get_embedding(question,engine="text-embedding-ada-002") # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model)

        blah = CogSearchHelper(self.index)
        relevant_data, count, lst_file_name, embeddings_tuples, lst_embeddings_text, metadata = blah.search(question)
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key, chunk_size=1536)

        if len(embeddings_tuples) == 0:
            return("Sorry, I don't know the answer to that question. Please try rephrasing your question, Cognitive Search did not provide documents")
    
        db = FAISS.from_embeddings(embeddings_tuples, embeddings, metadata)
        docs_db = db.similarity_search_by_vector(question_embedding, k = 4)

        MODEL = "gpt-35-turbo-16k" # options: gpt-35-turbo, gpt-35-turbo-16k, gpt-4, gpt-4-32k
        COMPLETION_TOKENS = 1000

        full_question = "system prompt " + question
        llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=COMPLETION_TOKENS)
        #you could change this chain type to use mapreduce or something else
        chain = load_qa_with_sources_chain(llm, chain_type="stuff" )
        response = chain({"input_documents": docs_db, "question": full_question, "language": "English", "existing_answer" : ""}, 
                         return_only_outputs=True)

        if response['output_text'] == "I don't know the answer to that question. Please try rephrasing your question.":
            chain = load_qa_with_sources_chain(llm, chain_type="stuff" )
            response = chain({"input_documents": docs_db, "question": "tell me about a " + question, "language": "English", "existing_answer" : ""}, 
                            return_only_outputs=True)
        

        return(response['output_text'])

 