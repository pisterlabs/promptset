import streamlit as st

st.header('My header')
import torch
import textwrap
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline

# Install the required libraries


from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModel
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Download the model (NousResearch's Llama2)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             load_in_4bit=True,
                                             bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.float16)

# Define Transformers pipeline
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                max_new_tokens=4956,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

# Define the prompt format
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
As the leader of a sizable team in a dynamic business, I'm tasked with improving our supply chain management process. Recently, we've been facing issues like increased costs, longer lead times, and decreased customer satisfaction, all of which we believe are interconnected. To address these challenges, I need your assistance in optimizing our supply chain management. Please provide insights, strategies, and best practices that can help us streamline our operations, reduce costs, improve efficiency, and ultimately enhance customer satisfaction. Additionally, consider the latest technologies and innovations that could be integrated into our supply chain to make it more agile and responsive to market demands. If you don't know the answer to a question, please don't share false information. Just say you don't know and you are sorry!"""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT, citation=None):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST

    if citation:
        prompt_template += f"\n\nCitation: {citation}"  # Insert citation here

    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")

def generate(text, citation=None):
    prompt = get_prompt(text, citation=citation)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 max_length=4956,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs

def parse_text(text):
    wrapped_text = textwrap.fill(text, width=100)
    print(wrapped_text + '\n\n')

# Defining Langchain LLM
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0.3, 'max_length': 4956, 'top_k': 50})

system_prompt = "You are an advanced supply chain optimization expert"
instruction = "Use the data provided to you to optimize the supply chain:\n\n {text}"
template = get_prompt(instruction, system_prompt)
print(template)

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)

import pandas as pd

# Assuming you have a CSV file named 'merged_data.csv'
df_supplier = pd.read_csv('merged_data.csv')

text = f"Based on the data provided how can you optimize my supply chain by prorviding me with the optmized solution as well as the techniques used. {df_supplier}"

response = llm_chain.run(text)
print(response)
