from dotenv import load_dotenv
from langchain.llms.base import LLM
from llama_index import (
    SimpleDirectoryReader,
    GPTListIndex,
    PromptHelper,
    LLMPredictor,
    ServiceContext
)
import torch
import time
import os
from transformers import pipeline

os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'

# llama_index

def timeit():
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            args = [str(arg) for arg in args]
            print(f'[{(end-start):.8f} seconds]: f({args}) -> {result}')
            return result
        return wrapper
    return decorator

load_dotenv()
prompt_helper = PromptHelper(
    max_input_size=1024,
    num_output=256,
    max_chunk_overlap=20,
)

class LocalOPT(LLM):
    model_name = 'facebook/opt-iml-1.3b'
    pipeline = pipeline('text-generation', 
                        model=model_name,
                        device="cuda:0",
                        model_kwargs={"torch_dtype": torch.bfloat16}
                        )
    
    def _call(self, prompt:str, stop=None) -> str:
        response = self.pipeline(prompt, 
                                 max_new_tokens=256)[0]["generated_text"]
        return response[len(prompt):]

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}
    
    @property
    def _llm_type(self):
        return "custom"
    
@timeit()    
def create_index():
    print("Creating index")
    llm = LLMPredictor(llm=LocalOPT())      
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm,
        prompt_helper=prompt_helper
    )
    docs = SimpleDirectoryReader('news').load_data()
    index = GPTListIndex.from_documents(docs, service_context=service_context)
    print("Done creating index")
    return index

def execute_query(question: str):
    print(f'"{question}"')
    response = index.query(
        question,
        exclude_keywords=["gas"],
        #required_keywords=["coal"]
        #response_mode="no_text"
    )
    return response

if __name__ == "__main__":
    filename = 'demo9.json'
    if not os.path.exists(filename):
        print("no local cache")
        index = create_index()
        index.save_to_disk(filename)
    else:
        print("loading local cache")
        index = GPTListIndex.load_from_disk(filename)
    response = execute_query("how to renerate SSR manifest.json file?")
    print(response)