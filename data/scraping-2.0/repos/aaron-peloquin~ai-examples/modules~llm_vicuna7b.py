from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch

model_path = ".\..\\ai-models\\vicuna-7B-1.1-HF"
tokenizer = LlamaTokenizer.from_pretrained(model_path)

base_model = LlamaForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map='auto',
)

pipe = pipeline(
    "text-generation",
    model=base_model,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    tokenizer=tokenizer,
    max_length=1152,
    temperature=0.34,
    top_p=0.95,
    repetition_penalty=1.2,
)

llm = HuggingFacePipeline(pipeline=pipe)
