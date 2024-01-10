from .fastapi_request import Request, VectorDBType
from .initialise import (load_vector_db_opensearch, 
                         sagemaker_endpoint_for_text_generation)
from fastapi import APIRouter
from typing import Any, Dict
import os
from langchain.chains.question_answering import load_qa_chain
from .query_llm import query_sm_endpoint
import boto3
import logging
from langchain import PromptTemplate


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()


_vector_db = None
_current_vectordb_type = None
_sm_llm = None

router = APIRouter()

ssm = boto3.client('ssm')
region = ssm.get_parameter(Name='REGION', WithDecryption=True)['Parameter']['Value']
opensearch_domain_endpoint = ssm.get_parameter(Name='OPENSEARCH_DOMAIN_ENDPOINT', WithDecryption=True)['Parameter']['Value']
opensearch_index = ssm.get_parameter(Name='OPENSEARCH_INDEX', WithDecryption=True)['Parameter']['Value']

def _init(req:Request):
    global _vector_db
    global _current_vectordb_type
    logger.info(f"req.vector_db_type: {req.vectordb_type}, _vector_db: {_vector_db}")
    if req.vectordb_type != _current_vectordb_type:
        logger.info(f"req.vectordb_type={req.vectordb_type} does not match _current_vectordb_type={_current_vectordb_type}, "
                    f"resetting _vector_db")
        _vector_db = None
    if req.vectordb_type == VectorDBType.opensearch and _vector_db is None:

        _vector_db = load_vector_db_opensearch(region,
                                               opensearch_domain_endpoint,
                                               opensearch_index,
                                               req.embeddings_generation_model_name)
    elif _vector_db is not None:
        logger.info(f"db already initialized, skipping")
    else:
        logger.error(f"req.vectordb_type={req.vectordb_type} which is not supported, _vector_db={_vector_db}")

    # similar to the above, but for the sagemaker endpoint
    global _sm_llm
    if _sm_llm is None:
        logger.info(f"SM LLM endpoint is not setup, setting it up")
        _sm_llm = sagemaker_endpoint_for_text_generation(req,region)
        logger.info("Sagemaker llm endpoint is now set up")
    else:
        logger.info(f"SM LLM endpoint is already setup, skipping")


@router.post("/text2text")
async def llm_text2text(req: Request) -> Dict[str, Any]:
    # debugdding request
    logger.info(f"req: {req}")

    # _init(req)

    answer = query_sm_endpoint(req)
    resp = {'question': req.query, 'answer': answer}
    return resp


@router.post("/rag")
async def llm_rag(req: Request) -> Dict[str, Any]:
    # debugdding request
    logger.info(f"req: {req}")

   # initialize the vector db and the sagemaker endpoint
   # it will be saved to gloabl  variable
    _init(req)

    # Use the vector db to find similar documents to the query
    # the vector db call would automatically convert the query text
    # into embeddings
    try:
        docs = _vector_db.similarity_search(req.query, k=req.max_matching_docs)
    except Exception as e:
        logger.error(f"error in similarity search, error={e}")
        raise e
    
    logger.info(f"there are the {req.max_matching_docs} closest documents to the query= \"{req.query}\"")
    for doc in docs:
        logger.info(f"--------")
        logger.info(f"doc: {doc}")
        logger.info(f"--------")
    
    #define prompt
    prompt_template = """Answer based on context:\n {context} \n Question: {question} \n Answer:"""
    prompt = PromptTemplate(
        template = prompt_template, input_variables = ["context", "question"]

    )
    logger.info(f"prompt sent to llm = \"{prompt}\"")
    # using load_qa_chain which is a high-level api than LLMChain 
    chain = load_qa_chain(llm=_sm_llm, prompt=prompt, chain_type="stuff")
    answer = chain({"input_documents": docs, "question": req.query}, return_only_outputs=True)['output_text']
    logger.info(f"answer received from llm, question: \"{req.query}\" answer: \"{answer}\"")
    resp  = {'question': req.query, 'answer': answer}
    if req.verbose:
        resp['docs'] = docs
    return resp


