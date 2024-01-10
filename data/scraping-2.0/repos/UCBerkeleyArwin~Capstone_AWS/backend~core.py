#%%writefile core.py


# App core features

from typing import Any, List, Dict

#Summary and checklist packages
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
# Chat packages
import torch
import os
# import runpod
from langchain.llms import HuggingFaceTextGenInference
from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.callbacks import streaming_stdout
from langchain import hub


# Global variables
# huggingfacehub_api_token = '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'
# zephyr_repo = 'HuggingFaceH4/zephyr-7b-beta'
rag_prompt = hub.pull("rlm/rag-prompt-mistral")


# TGI URL
# runpod.api_key = "O6FCUQVG8N3B58HLJZZ84P6DLNDRNQSU4KGANNG4"
# mistral_pod = runpod.get_pods()[0]
# mistral_pod
# inference_url = f"https://{mistral_pod['id']}-80.proxy.runpod.net"
inference_url = "https://wp021uax7a.execute-api.us-west-2.amazonaws.com/default/mistralrequest"


# Chat LLM
chat_llm = HuggingFaceTextGenInference(
    inference_server_url=inference_url,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)


# Summarization and checklist LLM
sum_check_llm = HuggingFaceTextGenInference(
    inference_server_url=inference_url,
    max_new_tokens=1012,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03
)


# Call LLM for summary
def run_llm_summarize(document_object: Any):

    docs = document_object

    # Map
    map_template = """<s> [INST] The following is a collection of excerpts from a compliance document:[/INST] </s>
    {docs}
    [INST] Based on the provided excerpts, summarize the main theme.
    Helpful Answer:[/INST]"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=sum_check_llm, prompt=map_prompt)

    # Reduce
    reduce_template = """<s> [INST] The following is set of summaries:[/INST] </s>
    {doc_summaries}
    [INST] Take these and distill it into a final, consolidated summary. Ensure the final output is concise and easy to read.
    Helpful Answer:[/INST]"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=sum_check_llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=1000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0)

    split_docs = text_splitter.split_documents(docs)

    summary = map_reduce_chain.run(split_docs)

    return summary


# Call LLM for checklist
def run_llm_checklist(document_object: Any):

    docs = document_object

    # Map
    map_template = """<s> [INST] The following is a collection of guidance from a compliance document:[/INST] </s>
    {docs}
    [INST] Based on the provided guidance, summarize a list of suggestions.
    Helpful Answer:[/INST]"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=sum_check_llm, prompt=map_prompt)

    # Reduce
    reduce_template = """<s> [INST] The following is a collection of suggestions from a compliance document:[/INST] </s>
    {doc_summaries}
    [INST] Take these and distill them into a final, consolidated list of suggestions. Ensure the final output is concise and easy to read.
    Helpful Answer:[/INST]"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=sum_check_llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=1000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0)

    split_docs = text_splitter.split_documents(docs)

    suggestion_list = map_reduce_chain.run(split_docs)

    return suggestion_list


# Call LLM for chat
def run_llm_chat(vector_database: Any, question: str):

    # Vector DB retriever
    retriever = vector_database.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 50})

    # QA Retriever
    qa = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # LLM Response
    output = qa({'query':question})

    # Response & source pages
    response = output["result"]
    source_page = [doc.metadata["page"] for doc in output["source_documents"]]

    return response, source_page
