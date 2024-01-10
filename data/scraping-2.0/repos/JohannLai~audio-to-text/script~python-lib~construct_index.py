from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display

def construct_index(file_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 1200
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # if exists, skip
    if os.path.exists(file_path + ".json"):
        print ("[Warning]" + file_path + ".json" + " already exists, skipping")
        return

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    file_paths = [file_path]

    documents = SimpleDirectoryReader(input_files=file_paths).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )


    # save with file name, remove extension, and add .json
    index.save_to_disk(file_path + ".json")

    return index

def ask_ai(filepath, query):
    index = GPTSimpleVectorIndex.load_from_disk(filepath)
    response = index.query(query, response_mode="compact")

    print (response.response)
    
def summary(filepath, query):
    index = GPTSimpleVectorIndex.load_from_disk(filepath)
    response = index.query(query, response_mode="compact")
    print (response)

if __name__ == "__main__":
    construct_index(sys.argv[1])
    summary(sys.argv[1] + ".json", sys.argv[2])
    