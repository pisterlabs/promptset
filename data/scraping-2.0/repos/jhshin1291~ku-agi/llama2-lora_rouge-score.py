#!/usr/bin/python

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
from trl import SFTTrainer
import re
import pdb

import evaluate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms  import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

import time


# [1] Setting Dataset & Basemodel
data_name = "ccdv/arxiv-summarization"
test_data = load_dataset(data_name, split="test")

# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-hf"
fine_tuned_model_name = "/home/work/data_yhgo/cyshin/agi/fine-tune_ccdv-sum_epoch01/"

# [2] Creating Llama2 Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# [3] Creating configuration for Quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# [4] Instantiating base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto" # accelerate library
    # device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

base_model_pipe = pipeline(task="text-generation", model=base_model, tokenizer=llama_tokenizer, max_new_tokens=128)
langchain_pipe = HuggingFacePipeline(pipeline=base_model_pipe)

# [6] Loading fine-tuned model
# fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_name, local_files_only=True)

# fine_tuned_model_pipe = pipeline(task="text-generation", model=fine_tuned_model, tokenizer=llama_tokenizer, max_new_tokens=128)
# langchain_pipe = HuggingFacePipeline(pipeline=fine_tuned_model_pipe)

# [7] Generating summurization logic for llama2 with text-generation task 
# LangChain: https://python.langchain.com/docs/get_started/introduction
def generate_summary_chunk(text_chunk):
  # Defining the template to generate summary
  template = """
  Write a concise summary of the text, return your responses with only 1 answer that cover the key points of the text.
  ```{text}```
  SUMMARY:
  """
  prompt = PromptTemplate(template=template, input_variables=["text"])
  llm_chain = LLMChain(prompt=prompt, llm=langchain_pipe)
  summary = llm_chain.run(text_chunk)
  # print("_________________")
  # print("text: ", text_chunk)
  # print("_________________")
  splited_summary = summary.split(".")
  if splited_summary[0] == "1":
    summary = splited_summary[1] + "."
  else:
    summary = splited_summary[0] + "."
  # print("summary: ", summary)
  return summary

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=10, length_function=len)

def generate_summary(input_text):
  chunks = text_splitter.split_text(input_text)
  print("input_text length: ", len(input_text))
  print("#chunks: ", len(chunks))
  chunk_summaries = []

  start = time.time()

  for chunk in chunks:
    summary = generate_summary_chunk(chunk)
    chunk_summaries.append(summary)      

  end = time.time()

  print(f"{end - start:.5f} sec")

  combined_summary = "\n".join(chunk_summaries) 
  return combined_summary


# [7] Predicting & evaluating rouge-score
references = []
predictions = []

num_paper = 10
cur_idx = 0

rouge = evaluate.load("rouge")
for item in test_data:
  reference = item['abstract']
  references.append(reference)

  prediction = generate_summary(item['article'])
  predictions.append(prediction)

  cur_idx += 1

  if cur_idx == num_paper:
    break
  
print(num_paper, "paper score")

results = rouge.compute(predictions=predictions,
                        references=references)

print(results)
