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

from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone


class GetPineconeVectorSemanticSearch(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        dependencies = ['langchain', 'openai', 'pinecone-client','tiktoken']
        version = '0.0.1-SNAPSHOT'
        description = 'Performs a similarity search in Pinecone based on the query/question that is asked and returns list of similar text docs with metadata.'
        tags = ['Pinecone','OpenAI', 'AI', 'Vector Similarity Search', 'Vector Database']


    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.openai_api_key = PropertyDescriptor(
            name="OpenAI API Key",
            description="The API key to connect to OpeanAI services",
            required = True,
            sensitive = True
        )
        self.openai_embedding_model = PropertyDescriptor(
            name="OpenAI Embedding Model",
            description="The OpenAI embedding model to use to convert query/question to a text embedding which is then used to search for similar docs.",
            required = True,
            default_value = "text-embedding-ada-002",
            allowable_values = ['text-embedding-ada-002', 'text-davinci-001', 'text-curie-001', 'text-babbage-001', 'text-ada-001']
        )

        self.pinecone_api_key = PropertyDescriptor(
            name="Pinecone API Key",
            description="The API key to connect to the Pinecone",
            required = True,
            sensitive = True
        )
        self.pinecone_environment_name = PropertyDescriptor(
            name="Pinecone Env Name",
            description="The environment that the Pinecone index is located.",
            required = True,
            sensitive = True
        )

        self.pinecone_index_name = PropertyDescriptor(
            name="Index Name",
            description="The Pinecone index to store the embeddings.",
            required = True,
        )

        self.pinecone_namespace = PropertyDescriptor(
            name="Namepace",
            description="The  namespace within the index to store the vector embeddings.",
            required = True,
        )

        self.search_results_size= PropertyDescriptor(
            name="Number of Similar Documents to Return",
            default_value="10",
            description="The number of similar documents to return from the similarity searech",
            required=True,
        )

        self.descriptors = [self.openai_api_key, self.openai_embedding_model, self.pinecone_api_key, self.pinecone_environment_name, self.pinecone_index_name, self.pinecone_namespace, self.search_results_size]

    def onScheduled(self, context):
        self.logger.info("Initializing OpenAI and Pinecone Services.")

        # Get the properties from the processor needed to configure the OpenAI Embedding Service
        openai_api_key = context.getProperty(self.openai_api_key.name).getValue()
        model_name = context.getProperty(self.openai_embedding_model.name).getValue()
        self.openai_embedding_service = OpenAIEmbeddings(openai_api_key=openai_api_key, model=model_name)

        # Initialize Pinecone and get the index we will be upserting into.
        pinecone_api_key =  context.getProperty(self.pinecone_api_key.name).getValue()
        pinecone_environment = context.getProperty(self.pinecone_environment_name.name).getValue()
        pinecone.init(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        pinecone_index = context.getProperty(self.pinecone_index_name.name).getValue()
        namespace = context.getProperty(self.pinecone_namespace.name).getValue()
        self.pinecone_index = Pinecone.from_existing_index(index_name=pinecone_index, embedding=self.openai_embedding_service, namespace=namespace)



    def transform(self, context, flowFile):

        query = flowFile.getContentsAsBytes().decode('utf-8')
        self.logger.info("Inside transform of GetPineconeVectorSemanticSearch with query["+query+"]")

        # Do Similartity Seaerch
        search_results_size = context.getProperty(self.search_results_size.name).asInteger()
        namespace = context.getProperty(self.pinecone_namespace.name).getValue()

        query_embedding = self.openai_embedding_service.embed_query(query)
        self.logger.info(f" The embedding for the query[{query} is {str(query_embedding)}")

        doc_results_from_search = self.pinecone_index.similarity_search(query=query, k=search_results_size, namespace=namespace)
        self.logger.info("Number of results returned by pinecone is: " + str(len(doc_results_from_search)))

        # Convert list of Documents into list of json strings
        search_results_json_list = [result.json() for result in doc_results_from_search]
        # Convert list of json strings into single json string to mimic what we return in NiFi flowfile result
        search_results_json_string = json.dumps(search_results_json_list)

        return FlowFileTransformResult(relationship="success", contents=search_results_json_string)


    def getPropertyDescriptors(self):
        return self.descriptors
