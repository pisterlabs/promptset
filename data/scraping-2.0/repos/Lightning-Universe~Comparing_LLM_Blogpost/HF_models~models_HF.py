import torch
from langchain.llms.base import LLM
from transformers import (T5Tokenizer, 
                        T5ForConditionalGeneration,
                        AutoTokenizer, 
                        pipeline, 
                        AutoModelForCausalLM,
                        GPTNeoXForCausalLM, 
                        GPTNeoXTokenizerFast)
from typing import List, Optional
from pydantic import BaseModel, Extra  

## "balanced_low_0" 
class CustomPipeline(LLM, BaseModel):
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def __init__(self, model_id):
        super().__init__()
        global model, tokenizer, model_name
        model_name = model_id 
        device_map = "auto"
        if model_id == "google/ul2" or model_id == "google/flan-t5-xxl":
            model = T5ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device_map)      
            if  model_id == "google/ul2":
                tokenizer = AutoTokenizer.from_pretrained("google/ul2")
            else:
                tokenizer = T5Tokenizer.from_pretrained(model_id)
        elif model_id == "facebook/opt-iml-max-30b":
            model =  pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype":torch.bfloat16}, device_map=device_map)
        elif model_id == "bigscience/bloomz-7b1":
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device_map, offload_folder="offload")  
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        elif model_id=="EleutherAI/gpt-j-6B":
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map) 
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        elif model_id == "EleutherAI/gpt-neox-20b":
            model = GPTNeoXForCausalLM.from_pretrained(model_id, device_map=device_map) 
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_id)
        elif model_id == "EleutherAI/pythia-12b-deduped":
            model = GPTNeoXForCausalLM.from_pretrained(
                        model_id,
                        revision="step3000",
                        device_map=device_map) 
            tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        revision="step3000",
                        )
        elif model_id == "dolly-v2-12":
            model = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
        elif model_id == "t5-11b":
            model = T5ForConditionalGeneration.from_pretrained("t5-11b",torch_dtype=torch.bfloat16, device_map=device_map)
            tokenizer = T5Tokenizer.from_pretrained("t5-11b")
        elif model_id== "ul2":
            model = T5ForConditionalGeneration.from_pretrained("google/ul2", torch_dtype=torch.bfloat16, device_map=device_map)                                                                                                  
            tokenizer = AutoTokenizer.from_pretrained("google/ul2")
        elif model_id == "OPT":
            model = AutoModelForCausalLM.from_pretrained("facebook/opt-66b", torch_dtype=torch.bfloat16, device_map=device_map)
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b", use_fast=False)
        elif model_id == "Cerebras-GPT-13B":
            tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B")
            model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-13B",torch_dtype=torch.bfloat16, device_map=device_map)
        elif model_id == "nomic-ai/gpt4all-j":
            model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", revision="v1.2-jazzy",torch_dtype=torch.bfloat16, device_map=device_map)
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            print(tokenizer)


        print(model_name)
    @property
    def _llm_type(self) -> str:
        return "custom_pipeline"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        if model_name == "facebook/opt-iml-max-30b" or model_name == "dolly-v2-12":
            prompt_length = len(prompt)
            response = model(prompt,max_new_tokens = 70)[0]["generated_text"]
            return response
        else:
            with torch.no_grad():
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
                outputs = model.generate(input_ids, max_new_tokens = 70)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response
