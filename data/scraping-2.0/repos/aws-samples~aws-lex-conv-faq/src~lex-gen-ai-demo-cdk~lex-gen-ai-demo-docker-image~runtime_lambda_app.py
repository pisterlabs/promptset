import boto3
from botocore.exceptions import ClientError
import logging
import json
import os
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
from llama_index import (
    LangchainEmbedding,
    PromptHelper,
    ResponseSynthesizer,
    LLMPredictor,
    ServiceContext,
    Prompt,
)

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.vector_stores.types import VectorStoreQueryMode
from llama_index import StorageContext, load_index_from_storage

s3_client = boto3.client('s3')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENDPOINT_NAME = "huggingface-pytorch-sagemaker-endpoint"
OUT_OF_DOMAIN_RESPONSE = "I'm sorry, but I am only able to give responses regarding the source topic"
INDEX_WRITE_LOCATION = "/tmp/index"
ACCOUNT_ID = boto3.client('sts').get_caller_identity().get('Account')
INDEX_BUCKET = "lexgenaistack-created-index-bucket-"+ACCOUNT_ID
RETRIEVAL_THRESHOLD = 0.4

# define prompt helper
max_input_size = 400  # set maximum input size
num_output = 50  # set number of output tokens
max_chunk_overlap = 0  # set maximum chunk overlap
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)


def handler(event, context):

    # lamda can only write to /tmp/
    initialize_cache()

    # define our LLM
    llm_predictor = LLMPredictor(llm=CustomLLM())
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(cache_folder="/tmp/HF_CACHE"))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embed_model,
    )

    ### Download index here
    if not os.path.exists(INDEX_WRITE_LOCATION):
        os.mkdir(INDEX_WRITE_LOCATION)
    try:
        s3_client.download_file(INDEX_BUCKET, "docstore.json", INDEX_WRITE_LOCATION + "/docstore.json")
        s3_client.download_file(INDEX_BUCKET, "index_store.json", INDEX_WRITE_LOCATION + "/index_store.json")
        s3_client.download_file(INDEX_BUCKET, "vector_store.json", INDEX_WRITE_LOCATION + "/vector_store.json")

        # load index
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_WRITE_LOCATION)
        index = load_index_from_storage(storage_context, service_context=service_context)
        logger.info("Index successfully loaded")
    except ClientError as e:
        logger.error(e)
        return "ERROR LOADING/READING INDEX"

    retriever = VectorIndexRetriever(
        service_context=service_context,
        index=index,
        similarity_top_k=5,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,  # doesn't work with simple
        alpha=0.5,
    )

    # configure response synthesizer
    synth = ResponseSynthesizer.from_args(
        response_mode="simple_summarize",
        service_context=service_context
    )

    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)
    query_input = event["inputTranscript"]

    try:
        answer = query_engine.query(query_input)
        if answer.source_nodes[0].score < RETRIEVAL_THRESHOLD:
            answer = OUT_OF_DOMAIN_RESPONSE
    except:
        answer = OUT_OF_DOMAIN_RESPONSE

    response = generate_lex_response(event, {}, "Fulfilled", answer)
    jsonified_resp = json.loads(json.dumps(response, default=str))
    return jsonified_resp

def generate_lex_response(intent_request, session_attributes, fulfillment_state, message):
    intent_request['sessionState']['intent']['state'] = fulfillment_state
    return {
        'sessionState': {
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Close'
            },
            'intent': intent_request['sessionState']['intent']
        },
        'messages': [
            {
                "contentType": "PlainText",
                "content": message
            }
        ],
        'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
    }

# define prompt template
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "CONTEXT1:\n"
    "{context_str}\n\n"
    "CONTEXT2:\n"
    "CANNOTANSWER"
    "\n---------------------\n"
    'Given this context, please answer the question if answerable based on on the CONTEXT1 and CONTEXT2: "{query_str}"\n; '  # otherwise specify it as CANNOTANSWER
)
my_qa_template = Prompt(template)

def call_sagemaker(prompt, endpoint_name=ENDPOINT_NAME):
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            # "top_p": 0.9,
            "temperature": 0.1,
            "max_new_tokens": 200,
            "repetition_penalty": 1.03,
            "stop": ["\nUser:", "<|endoftext|>", "</s>"]
        }
    }

    sagemaker_client = boto3.client("sagemaker-runtime")
    payload = json.dumps(payload)
    response = sagemaker_client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=payload
    )
    response_string = response["Body"].read().decode()
    return response_string

def get_response_sagemaker_inference(prompt, endpoint_name=ENDPOINT_NAME):
    resp = call_sagemaker(prompt, endpoint_name)
    resp = json.loads(resp)[0]["generated_text"][len(prompt):]
    return resp

class CustomLLM(LLM):
    model_name = "tiiuae/falcon-7b-instruct"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = get_response_sagemaker_inference(prompt, ENDPOINT_NAME)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"
    
def initialize_cache():
    if not os.path.exists("/tmp/TRANSFORMERS_CACHE"):
        os.mkdir("/tmp/TRANSFORMERS_CACHE")

    if not os.path.exists("/tmp/HF_CACHE"):
        os.mkdir("/tmp/HF_CACHE")