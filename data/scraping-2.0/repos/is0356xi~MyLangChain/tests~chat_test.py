from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Prompts: プロンプトを作成
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# モデルのダウンロード
model_id = "jquave/gpt4all-lora-unfiltered-quantized"
#model_id = "andreaskoepf/pythia-1.4b-gpt4all-pretrain"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)