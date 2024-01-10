#langchain imports
import langchain
from langchain.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
#regular imports
import os
import argparse

def get_documents():
    alarms_json_by_receiver = os.path.join('/data', 'raw', 'alarms', 'by_receiver')
    loader = DirectoryLoader(alarms_json_by_receiver, glob='**/*.json', show_progress=True, loader_cls=JSONLoader, jq_schema=".alerts[].annotations.description")
    documents = loader.load()
    return documents

def get_document(doc_path="/data/raw/alarms/by_receiver/aris.json"):
    loader = JSONLoader(file_path=doc_path, jq_schema=".alerts[].annotations.description")
    data = loader.load()
    return data

def get_model(model_id):
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        model_kwargs={"temperature": 0, "max_new_tokens": True, "load_in_8bit":True},
)
    return llm

def get_llm_chain(prompt, llm):
    llm_chain = LLMChain(llm = llm, prompt = prompt)
    return llm_chain

def get_prompt():
    #mention the access to the json files
    #mention that it needs to write a short answer that summarizes the json file
    #mention that the data is about alarms that are can be caused by ...,...,...
    template = """
            based on this json document, summarize the issue that caused the alarm in bullet points.
            """
    prompt = PromptTemplate(template = template, input_variables = [])
    return prompt

def summarize_document(document, llm_chain):
    return llm_chain.run(input_documents = document)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    document = get_document()
    print(document)
    exit()
    #load the model
    model_id = 'bigscience/bloom-1b7'
    local_llm = get_model(model_id)

    #load the documents
    #documents = get_documents()
    #test_doc = documents[0]

    #create the prompt for each document
    #TODO: add memory to the prompt of alarms that already have been summarized
    prompt = get_prompt()
    llm_chain = get_llm_chain(prompt, local_llm)
    answers = []
    answers.append(summarize_document(document, llm_chain))
    #for document in documents:
    #    answers.append(summarize_document(document, llm_chain))
    print(answers)

