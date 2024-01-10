import requests
import json
from credentials import * 

import json
import numpy as np
import os

from langchain.llms import AzureOpenAI
#from langchain import FAISS
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
from transformers import GPT2TokenizerFast
import pandas as pd


class CogSearchHelper:
    def __init__(self, index):
        self.service_name = creds['COG_SEARCH_RESOURCE']
        self.search_key = creds['COG_SEARCH_KEY']
        self.storage_connectionstring = creds['STORAGE_CONNECTION_STRING']
        self.storage_container = creds['STORAGE_CONTAINER']
        self.cognitive_service_key = creds['COG_SERVICE_KEY']
        if index == None:
            self.index = creds['COG_SEARCH_INDEX']
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

    def search_semantic(self, question):
        print('searching semantic')

        response = openai.Embedding.create(input=question,engine="text-embedding-ada-002")
        q_embeddings = response['data'][0]['embedding']
        
        if len(question) > 0:
            endpoint = "https://{}.search.windows.net/".format(self.service_name)
            url = '{0}indexes/{1}/docs/search?api-version=2021-04-30-Preview'.format(endpoint, self.index)

            print(url)

            payload = json.dumps({
            "search": question,
            "queryType": "semantic",
            "queryLanguage": "en-us",
            "captions": "extractive",
            "answers": "extractive",
            "semanticConfiguration": "semanic-config",
            "count": True,
            })
            headers = {
            'api-key': '{0}'.format(self.search_key),
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)
            
            obj = response.json()

            try:
                answer = obj['@search.answers'][0]['text']
            except:
                answer = obj['value'][0]['@search.captions'][0]['text']
                pass

            relevant_data = []
            lst_embeddings_text = []
            lst_embeddings = []
            lst_file_name = []
            count = 0
            #should only grab 1 from each document.
            for x in obj['value']:
                if x['@search.rerankerScore'] > 0.5:
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

                    #get the most relevant embedding and the most relevant text for the document
                    df = pd.DataFrame(list(zip(embeddings_text, curie_list)),columns =['text', 'embedding_values'])
                    df["similarities"] = df.embedding_values.apply(lambda x: cosine_similarity(x, q_embeddings))
                    res = (df.sort_values("similarities", ascending=False).reset_index(drop=True).head(1))

                    embedding_text_most_relevant = res['text'][0]
                    embedding_vector_most_relevant = res['embedding_values'][0]

                    
                    # print('embedding_text_most_relevant = ' + embedding_text_most_relevant)
                    # print('embedding_vector_most_relevant = ' + str(embedding_vector_most_relevant))

                    lst_embeddings_text.append(embedding_text_most_relevant)
                    lst_embeddings.append(embedding_vector_most_relevant)
                    lst_file_name.append(file_name)

                    # for i in range(len(embeddings)):
                    #     lst_embeddings_text.append(embeddings_text[i])
                    #     lst_embeddings.append(np.fromstring(embeddings[i][1:-1], dtype=float, sep=','))
                    #     lst_file_name.append(file_name)
                

            tuples_list = []
            tokencount = 0
            for i in range(len(lst_embeddings_text)):
                tuples_list.append((lst_embeddings_text[i], lst_embeddings[i]))

            # print('tuples_list = ' )
            # print(tuples_list)

            return answer, relevant_data, count, lst_file_name, tuples_list, lst_embeddings_text
        
    def create_datasource(self):
        endpoint = "https://{}.search.windows.net/".format(self.service_name)


        url = '{0}/datasources/{1}-datasource?api-version=2020-06-30'.format(endpoint, self.index)

        print(url)
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
            return response, False
        
    def create_index(self):

        endpoint = "https://{}.search.windows.net/".format(self.service_name)
        url = '{0}/indexes/{1}/?api-version=2021-04-30-Preview'.format(endpoint, self.index)
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
        "semantic": {
            "defaultConfiguration": None,
            "configurations": [
            {
                "name": "semanic-config",
                "prioritizedFields": {
                "titleField": {
                    "fieldName": "metadata_storage_name"
                },
                "prioritizedContentFields": [
                    {
                    "fieldName": "merged_content"
                    }
                ],
                "prioritizedKeywordsFields": [
                    {
                    "fieldName": "keyphrases"
                    },
                    {
                    "fieldName": "people"
                    },
                    {
                    "fieldName": "locations"
                    }
                ]
                }
            }
            ]
        },
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
    def create_skillset(self):
        endpoint = "https://{}.search.windows.net/".format(self.service_name)
        appfunctionurl = creds['APP_FUNCTION_URL']
        print(appfunctionurl)
        url = '{0}/skillsets/{1}-skillset?api-version=2021-04-30-Preview'.format(endpoint, self.index)
        print(url)
        payload = json.dumps({
        "@odata.context": "https://mmx-cog-search.search.windows.net/$metadata#skillsets/$entity",
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
    "uri": appfunctionurl,
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
            },
            {
            "name": "filename",
            "source": "/document/metadata_storage_name"
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
            "description": "/subscriptions/b071bca8-0055-43f9-9ff8-ca9a144c2a6f/resourceGroups/mmx-cognitive-services-rg/providers/Microsoft.CognitiveServices/accounts/xmm-cognitive-services",
            "key": "{0}".format(self.cognitive_service_key)
        },
        "knowledgeStore": None,
        "encryptionKey": None
        })
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': '{0}'.format(self.search_key)
        }

        
        response = requests.request("PUT", url, headers=headers, data=payload)

        

        if response.status_code == 201 or response.status_code == 204:
            return response, True
        else:

            return response, False

    def create_indexer(self):
        endpoint = "https://{}.search.windows.net/".format(self.service_name)
        url = '{0}/indexers/{1}-indexer/?api-version=2021-04-30-Preview'.format(endpoint, self.index)
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

    def run_indexer(self):
        endpoint = "https://{}.search.windows.net/".format(self.service_name)
        url = '{0}/indexers/{1}/run?api-version=2021-04-30-Preview'.format(endpoint, self.index + '-indexer')
        headers = {
        'api-key': self.search_key,
        'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers)
        print(response.text)
        
class OpenAIHelper:
    def __init__(self, index):
        self.question_template = creds['QUESTION_TEMPLATE']
        if index == None:
            self.index = creds['COG_SEARCH_INDEX']
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
    
    def get_Answer(self, question):
        print('Get Answer')

        openai.api_type = "azure"
        openai.api_base = creds['AZURE_OPENAI_ENDPOINT']
        openai.api_version = "2022-12-01"
        os.environ['OPENAI_API_KEY'] = creds['AZURE_OPENAI_KEY']
        openai.api_key = os.getenv("OPENAI_API_KEY")

        from openai.embeddings_utils import get_embedding, cosine_similarity
        question_embedding = get_embedding(question,engine="text-embedding-ada-002") # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model)
        print(question_embedding)

        blah = CogSearchHelper(self.index)
        answer, relevant_data, count, lst_file_name, embeddings_tuples, lst_embeddings_text = blah.search_semantic(question)
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key, chunk_size=1536)

        full_question = creds['QUESTION_TEMPLATE'].format(question = question)

        print('full questoin = ' + full_question)

        print('relevant files:')
        for x in lst_file_name:
            print(x)

        print(embeddings_tuples)
        if len(embeddings_tuples) == 0:
            return("Sorry, I don't know the answer to that question. Please try rephrasing your question.")
    
        db = FAISS.from_embeddings(embeddings_tuples, embeddings)
        docs_db = db.similarity_search_by_vector(question_embedding, k = 4)



        #indexxtg3, and map reduce.
        if    self.get_the_token_count(full_question) + 100 < 3096:
                print("running stuff....")
                llm = AzureOpenAI(deployment_name=creds['TEXT_DAVINCI'], model_name="text-davinci-003", temperature=0.0, max_tokens=1000) 
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain({"input_documents": docs_db, "question": full_question, "language": "English", "existing_answer" : ""}, return_only_outputs=True)
        else:
                print("running a map reduce....")
                llm = AzureOpenAI(deployment_name=creds['TEXT_DAVINCI'], model_name="text-davinci-003", temperature=0.0, max_tokens=1000) 
                chain = load_qa_chain(llm, chain_type="map_reduce")
                response = chain({"input_documents": docs_db, "question": full_question, "language": "English", "existing_answer" : ""}, return_only_outputs=True)
        return(response['output_text'])

  

    


    def get_FollowUpAnswer(self, question, new_docsearch, lst_file_name):
        docs_db = new_docsearch.similarity_search(question)
        full_question = self.question_template.format(question, lst_file_name)
        llm = AzureOpenAI(deployment_name=creds['TEXT_DAVINCI'], model_name="text-davinci-003", temperature=0.0, max_tokens=2000) 
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain({"input_documents": docs_db, "question": full_question, "language": "English", "existing_answer" : ""}, return_only_outputs=True)
        return(response['output_text'])
