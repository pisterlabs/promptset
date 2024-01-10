from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
# from IPython.display import Markdown, display

os.environ["OPENAI_API_KEY"]="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"

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index
 


def ask_ai():
   
    while True: 
        index2 = GPTSimpleVectorIndex.load_from_disk('index.json')
        query = input("What do you want to ask? ")
        response = index2.query(query)
        # //bot emoji
            
        print(f"Response: {response.response}")
        # display(Markdown(f"Response: <b>{response.response}</b>"))

construct_index("context_data/data")
ask_ai()