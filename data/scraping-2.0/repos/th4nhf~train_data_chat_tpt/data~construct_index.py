from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os

def construct_index(src_path, out_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 512
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(src_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk(f"{out_path}/index.json")

    return index

if __name__ == "__main__":
    import os 
    src_path = os.getcwd() 
    dir_path = src_path + "/clean"
    out_path = src_path
    os.environ["OPENAI_API_KEY"] = "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"
    index = construct_index(src_path, out_path)