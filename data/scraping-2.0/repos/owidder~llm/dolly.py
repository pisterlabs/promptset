import torch
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
from transformers import pipeline
from typing import Optional, List, Mapping, Any
import os
from datetime import datetime

# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

class CustomLLM(LLM):
    model_name = "jeffwan/vicuna-13b"
    #model_name = "databricks/dolly-v2-12b"
    #model_name = "s-JoL/Open-Llama-V1"
    #model_name = "decapoda-research/llama-30b-hf"
    #model_name = "google/flan-ul2"
    pipeline = pipeline("text-generation", model=model_name, device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16}, trust_remote_code=True)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


# define our LLM
llm_predictor = LLMPredictor(llm=CustomLLM())
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
documents = SimpleDirectoryReader('./data').load_data()
index = GPTListIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()
print(datetime.now())
response = query_engine.query("What is a knotterbex policy?")
print(response)
print(datetime.now())
response = query_engine.query("Is it efficient?")
print(response)
print(datetime.now())