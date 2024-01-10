# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from nifiapi.properties import PropertyDescriptor
from nifiapi.properties import StandardValidators
from nifiapi.properties import ExpressionLanguageScope
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

from langchain.embeddings import OpenAIEmbeddings

class GetOpenAiVectorEmbedding(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        dependencies = ['langchain', 'openai','tiktoken']
        version = '0.0.1-SNAPSHOT'
        description = 'Creates text embeddings for each text chunk using OpeanAI embedding model services using langchain libraries'
        tags = ['AI', 'OpenAI',  'Embeddings', 'Langchain', 'Vectors' ]


    def __init__(self, **kwargs):
        # Build Property Descriptors

        self.openai_api_key = PropertyDescriptor(
            name="OpenAI API Key",
            description="The API key to connect to OpeanAI services",
            required = True,
            sensitive = True
        )
        self.openai_embedding_model = PropertyDescriptor(
            name="OpenAI Embedding Models",
            description="The OpenAI embedding model to use when creating the text embedding vector.",
            required = True,
            default_value = "text-embedding-ada-002",
            allowable_values = ['text-embedding-ada-002', 'text-davinci-001', 'text-curie-001', 'text-babbage-001', 'text-ada-001']
        )
        self.chunk_size = PropertyDescriptor(
            name="Chunk Size",
            description="The number of characters that each text chunk when used by OpenAI to create the text embedding",
            default_value = 1000,
            required = True,
        )

        self.descriptors = [self.openai_api_key, self.openai_embedding_model, self.chunk_size ]
        self.openai_embedding_service = None

    def onScheduled(self, context):
        self.logger.info("Initializing OpenAI Embedding Service")

        # Get the properties from the processor needed to configure the OpenAI Embedding model
        openai_api_key = context.getProperty(self.openai_api_key.name).getValue()
        model_name = context.getProperty(self.openai_embedding_model.name).getValue()

        # Initialize OpenAI Embedding Service
        self.openai_embedding_service = OpenAIEmbeddings(openai_api_key=openai_api_key, model=model_name)


    def transform(self, context, flowFile):
        self.logger.info("Inside transform of GetOpenAiVectorEmbedding..")

        # Convert the single json string into List of documents of type Dict
        chunked_docs_string = flowFile.getContentsAsBytes().decode('utf-8')
        chunk_docs_json_list_deserialized = json.loads(chunked_docs_string)

        self.logger.info("The number of text documents to be embedded are: " + str(len(chunk_docs_json_list_deserialized)))

        # Store the text and metadata for each text document in their own lists to pass to OpenAI
        texts = []
        metadatas = []
        for doc_dict in chunk_docs_json_list_deserialized:
            #doc_dict = json.loads(doc)
            texts.append(doc_dict['page_content'])
            metadata = {"title": doc_dict['metadata_title'],
                        "created": doc_dict['metadata_created'],
                        "updated": doc_dict['metadata_updated'],
                        "tags": doc_dict['metadata_tags'],
                        "source": doc_dict['metadata_source']
                        }
            metadatas.append(metadata)

        # Create an embedding for each text block
        chunk_size = context.getProperty(self.chunk_size.name).asInteger()
        vector_embeddings = self.openai_embedding_service.embed_documents(texts=texts, chunk_size=chunk_size)

        # Now that we have the embeddings, lets create list of json elements with text, metadata and vector embedding
        json_list_with_text_embeddings = []
        for text, vector_embedding, metadata in zip(texts, vector_embeddings, metadatas):
            text_embedding_json = {"text": text, "embedding": vector_embedding, "metadata": metadata}
            json_list_with_text_embeddings.append(text_embedding_json)

        # Convert the list of json strings into a single json string
        json_embedding_string = json.dumps(json_list_with_text_embeddings)

        return FlowFileTransformResult(relationship="success", contents=json_embedding_string)


    def getPropertyDescriptors(self):
        return self.descriptors
