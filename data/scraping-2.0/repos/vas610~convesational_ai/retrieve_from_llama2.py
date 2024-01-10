#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.vectorstores import FAISS

# Get Env Variables


load_dotenv()  # load the values for environment variables from the .env file

AWS_REGION = os.environ.get("AWS_REGION")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
LLAMA2_ENDPOINT = os.environ.get("LLAMA2_ENDPOINT")
MAX_HISTORY_LENGTH = os.environ.get("MAX_HISTORY_LENGTH")


def build_chain():

    # Sentence transformer
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Laod Faiss index
    db = FAISS.load_local("faiss_index", embeddings)

    # Default system prompt for the LLamav2 on SageMaker Jumpstart Endpoint
    system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    # Custom ContentHandler to handle input and output to the SageMaker Endpoint
    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            payload = {
                "inputs": [
                    [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": prompt},
                    ],
                ],
                "parameters": {"max_new_tokens": 1000, "top_p": 0.9, "temperature": 0.6},
            }
            input_str = json.dumps(
                payload,
            )
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            content = response_json[0]["generation"]["content"]
            return content

    # Langchain chain for invoking SageMaker Endpoint
    llm = SagemakerEndpoint(
        endpoint_name=LLAMA2_ENDPOINT,
        region_name=AWS_REGION,
        content_handler=ContentHandler(),
        # credentials_profile_name="credentials-profile-name", # AWS Credentials profile name 
        # callbacks=[StreamingStdOutCallbackHandler()],
        endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    )

    def get_chat_history(inputs) -> str:
        res = []
        for _i in inputs:
            if _i.get("role") == "user":
                user_content = _i.get("content")
            if _i.get("role") == "assistant":
                assistant_content = _i.get("content")
                res.append(f"user:{user_content}\nassistant:{assistant_content}")
        return "\n".join(res)

    condense_qa_template = """
    Given the following conversation and a follow up question, rephrase the follow up question
    to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    standalone_question_prompt = PromptTemplate.from_template(
        condense_qa_template,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        condense_question_prompt=standalone_question_prompt,
        return_source_documents=True,
        get_chat_history=get_chat_history,
        # verbose=True,
    )
    return qa


def run_chain(chain, prompt: str, history=[]):
    return chain({"question": prompt, "chat_history": history})
