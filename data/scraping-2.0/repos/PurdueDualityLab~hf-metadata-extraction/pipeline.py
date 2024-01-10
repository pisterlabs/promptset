import openai
from collections import Counter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
import time
import json
import argparse
import asyncio
import sys
import os

from utils import hf_utils
import keys
import prompt

os.environ["OPENAI_API_KEY"] = keys.OPENAI_API_KEY

parser = argparse.ArgumentParser(description="Python Script with Different Modes")
parser.add_argument("--data", choices = ["input", "filtered"], required= True, help = "Select to run through input data set or filtered model set")
parser.add_argument("--start", type = int, help= "Select which model index to start from in filtered_models.json file")
parser.add_argument("--range", type = int, help= "Select number of models to run through")
args = parser.parse_args()

if args.data == "input":
    with open("input.json", 'r') as json_file:
        data = json.load(json_file)

if args.data == "filtered":
    if args.start is None or args.range is None:
        parser.error("for filtered data, starting index and range required.")
    with open("filtered_models.json", 'r') as json_file:
        data = json.load(json_file)

with open("metaSchema.json", 'r') as json_file:
    schema = json.load(json_file)

with open("log.txt", 'w') as the_log:
    the_log.write("")
    the_log.close()

def get_data_schema() -> dict:

    simple_metadata_schema = {"simple":{"properties": {**schema["simple_metadata"]}}}
    complex_metadata_schema = {"complex":{"properties": {**schema["complex_metadata"]}}}
    return {**simple_metadata_schema, **complex_metadata_schema}

def pretty_print_docs(docs, metadata):
    return f"{'-' * 20} {metadata} {'-' * 20}\n\n" + f"\n{'-' * 30}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]) + "\n\n"

def log(text: str):
    with open("log.txt", 'a') as the_log:
        if(type(text) == str):
                the_log.write(text)
        elif(type(text) == dict):
            for key, value in text.items():
                the_log.write(f"{key}: {value}\n")
        the_log.close()

def log_list(content: str):
    line_width = 100
    lines = [content[i:i+line_width] for i in range(0, len(content), line_width)]
    with open("log.txt", "a") as file:
        for line in lines:
            file.write(f"{line.ljust(line_width)} \n")
        file.close()
    

log("Metadata Prompt: \n\n")
log(prompt.METADATA_PROMPT)
log("\nExtraction Prompts: \n\n" + str(prompt.SIMPLE_METADATA_EXTRACTION_PROMPT) + str(prompt.COMPLEX_METADATA_EXTRACTION_PROMPT) + "\n")

if args.data == "input":
    models_iterable = data

if args.data == "filtered":
    data = data.keys()
    start_index = args.start
    end_index = start_index + args.range
    models_iterable = data[start_index: end_index]

start_time = time.time()
final_result = {}

extraction_prompt = {
    "simple": ChatPromptTemplate.from_messages(
        [
            SystemMessage(content = (prompt.SIMPLE_METADATA_EXTRACTION_PROMPT)),
            HumanMessagePromptTemplate.from_template("{documents}"),
        ]
    ),
    "complex": ChatPromptTemplate.from_messages(
        [
            SystemMessage(content = (prompt.COMPLEX_METADATA_EXTRACTION_PROMPT)),
            HumanMessagePromptTemplate.from_template("{documents}"),
        ]
    ),
}      

#change between filtered_models and input_models
for model in models_iterable:
    model_result = {}
    # try:
    card = hf_utils.load_card(model)
    card = card.replace('"', "'")
    log(f"\n#####################{model}########################\n\n")
    model_result["domain"], model_result["model_tasks"] = hf_utils.get_domain_and_task(model)
    model_result["frameworks"]= hf_utils.get_frameworks(model)
    model_result["libraries"]= hf_utils.get_libraries(model)
    data_schema = get_data_schema()

    headers_to_split_on = [("#", "header 1"), ("##", "header 2"), ("###", "header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(card)
    #print(md_header_splits)

    vector_store = FAISS.from_documents(md_header_splits, OpenAIEmbeddings(allowed_special={'<|endoftext|>', '<|prompter|>', '<|assistant|>'}))
    llm = OpenAI(temperature = 0)
    chatbot = ChatOpenAI(temperature = 0.1, model = "gpt-3.5-turbo")


    for key in data_schema.keys():
        compressed_docs = ""
        for metadata in data_schema[key]["properties"]:
            retriever = vector_store.as_retriever(search_kwargs = {"k": 3})
            retriever_prompt = prompt.METADATA_PROMPT[metadata]
            #log(pretty_print_docs(docs, metadata))
            compressor = LLMChainExtractor.from_llm(llm)
            compression_retriever = ContextualCompressionRetriever(base_compressor = compressor, base_retriever = retriever)
            single_compressed_docs = compression_retriever.get_relevant_documents(retriever_prompt + f" Keep surrounding context in compressed document.")
            compressed_docs += pretty_print_docs(single_compressed_docs, "compressed" + metadata)
            log(pretty_print_docs(single_compressed_docs, "compressed " + metadata))    
        chain = create_extraction_chain(schema = data_schema[key], llm = chatbot, prompt = extraction_prompt[key])
        extraction_result = chain.run({
            "domain": model_result["domain"],
            "model": model,
            "documents": compressed_docs
            })
    
        if type(extraction_result) == list:
            model_result = {**model_result,**extraction_result[0]}
        if type(extraction_result) == dict:
            model_result = {**model_result,**extraction_result}

    
    print(model_result)
    log_list("\n" + str(model_result))
    final_result[model] = model_result

    #exceed token limit of 4096
    # except Exception as e:

    #     log(f"\nERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n {str(e)} \n")
    #     final_result[model] = str(e)
        

    #For eval purposes
    with open("result.json", "w") as json_file:
        json.dump(final_result, json_file, indent = 4)


# file_path = "result.json"
# with open(file_path, "w") as json_file:
#     json.dump(result, json_file, indent = 4)

end_time = time.time()
log(f"total elapsed time: {int((end_time - start_time)/3600)} hours {int((end_time-start_time)/60%60)} minutes {int(end_time-start_time)%60} seconds")
