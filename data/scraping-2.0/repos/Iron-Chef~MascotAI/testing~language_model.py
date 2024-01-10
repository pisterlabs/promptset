from transformers import AutoTokenizer
from langchain import HuggingFacePipeline

import torch
import transformers

def load_language_model(model_name):

    model = f"IronChef/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=500,
        do_sample=True,
        top_k=10,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipeline)

    return llm