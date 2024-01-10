import os
import json
import boto3
import logging
from typing import List, Callable
from urllib.parse import urlparse
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import SagemakerEndpointEmbeddings
from .fastapi_request import Request, sagemaker_endpoint_mapping, EmbeddingsModelName
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import time
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection
from langchain import SagemakerEndpoint


logger = logging.getLogger(__name__)

#  fetch value from aws parameter store

ssm = boto3.client('ssm')
region = ssm.get_parameter(Name='REGION', WithDecryption=True)['Parameter']['Value']
access_key = ssm.get_parameter(Name='ACCESS_KEY', WithDecryption=True)['Parameter']['Value']
secret_key = ssm.get_parameter(Name='SECRET_KEY', WithDecryption=True)['Parameter']['Value']
service = 'es'

aws4auth = AWS4Auth(access_key, secret_key, region, service)

class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(
            self, texts: List[str], 
            chunk_size: int = 5
    ) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size
        st = time.time()
        for i in range(0, len(texts), _chunk_size):
            response = self._embedding_func(texts[i:i + _chunk_size])
            results.extend(response)
        time_taken = time.time() - st
        logger.info(f"Embedding completes and it took {time_taken} seconds")
        return results

# class for serializing/deserializing requests/responses to/from the embeddings model
class ContentHandlerForEmbeddings(EmbeddingsContentHandler):
    """
    encode input string as utf-8 bytes, read the embeddings
    from the output
    """
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> List[str]:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["embedding"]
        if len(embeddings) == 1:
            return [embeddings[0]]
        return embeddings

# class for serializing/deserializing requests/responses to/from the llm model
class ContentHandlerForTextGeneration(LLMContentHandler):
    """
    encode input string as utf-8 bytes, read the text generation 
    from the output
    """
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]


# create the embedding
def _create_sagemaker_embeddings(endpoint_name: str, region: str) -> SagemakerEndpointEmbeddingsJumpStart:
    """
    Args:
        endpoint_name: The name of the Sagemaker Inference Endpoint.
        region: The region of the Sagemaker Inference Endpoint.
    """

    # create a content handler object which knows how to serialize
    # and deserialize communication with the model endpoint

    content_handler_for_embeddings = ContentHandlerForEmbeddings()

    # Sagemakder endpoint that will be used to embed query

    embeddings = SagemakerEndpointEmbeddingsJumpStart(
        endpoint_name=endpoint_name,
        region_name=region,
        content_handler=content_handler_for_embeddings
    )
    logger.info(f"embeddings type={type(embeddings)}")
    return embeddings

# loading vector store
def load_vector_db_opensearch(region:str,
                              opensearch_endpoint:str,
                              opensearch_index:str,
                              embedding_model_name:str) -> OpenSearchVectorSearch:
    logger.info(f"load_vector_db_opensearch, region={region}, "
                f"opensearch_domain_endpoint={opensearch_endpoint}, opensearch_index={opensearch_index}, "  
                f"embeddings_model={embedding_model_name}")
    

    embedding_model_name_enum = EmbeddingsModelName(embedding_model_name)
    embeddings_model_endpoint = sagemaker_endpoint_mapping[embedding_model_name_enum]
    logger.info(f"embeddings_model_endpoint={embeddings_model_endpoint}")

    embedding_function = _create_sagemaker_embeddings(embeddings_model_endpoint, region)
   
    vector_db = OpenSearchVectorSearch(index_name = opensearch_index,
                                       embedding_function = embedding_function,
                                       opensearch_url = opensearch_endpoint,
                                       timeout = 300,
                                       use_ssl = True,
                                       verify_certs = True,
                                       connection_class = RequestsHttpConnection,
                                       http_auth=aws4auth)
    logger.info(f"returning handle to OpenSearchVectorSearch, vector_db={vector_db}")
    return vector_db

# Sagemaker endpoint instanee for text generation
def sagemaker_endpoint_for_text_generation(req: Request, region:str) -> SagemakerEndpoint:
    parameters = {
        "max_length": req.max_length,
        "num_return_sequences": req.num_return_sequences,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "do_sample": req.do_sample,
        "temperature": req.temperature}
    logger.info(f"setting up llm text generation endpoint with parameters={parameters}")
    content_handler = ContentHandlerForTextGeneration()
    text_generation_endpoint_name = sagemaker_endpoint_mapping[req.text_generation_model_name]
    logger.info(f"text_generation_endpoint_name is: {text_generation_endpoint_name}")
    sm_llm = SagemakerEndpoint(
        endpoint_name=text_generation_endpoint_name,
        region_name = region,
        model_kwargs =  parameters,
        content_handler = content_handler)
    return sm_llm