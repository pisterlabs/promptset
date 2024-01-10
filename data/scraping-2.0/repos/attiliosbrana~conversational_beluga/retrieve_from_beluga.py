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
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

# Get Env Variables

load_dotenv()  # load the values for environment variables from the .env file

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
HF_MODEL = os.environ.get("HF_MODEL")
MAX_HISTORY_LENGTH = os.environ.get("MAX_HISTORY_LENGTH")

def build_chain():

    # Sentence transformer
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Load Faiss index
    db = FAISS.load_local("./faiss_index/", embeddings)

    model = HF_MODEL 
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    pipeline_ = pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=4000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline = pipeline_, model_kwargs = {'temperature':0.6, "top_p": 0.9})

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
        verbose=True,
    )
    return qa

def run_chain(chain, prompt: str, history=[]):
    return chain({"question": prompt, "chat_history": history})
