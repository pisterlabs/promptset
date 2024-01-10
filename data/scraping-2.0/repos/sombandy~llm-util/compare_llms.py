#!/usr/bin/env python

# stdlib
import json

# third party
import time
from dotenv import load_dotenv


load_dotenv()

from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.chat_models import ChatOpenAI

question = "Name five companies who offer similar products to Microsoft."

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])


def run_llm(llm, prompt, model_name):
    print("LLM model: ", model_name)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    s = time.time()
    response = llm_chain.run(question)
    e = time.time()
    elapsed = "{:.3f}".format(e - s)
    print("Time taken", elapsed)

    return {"response": response, "time": elapsed}


results = dict()

model = "gpt-3.5-turbo"
chat = ChatOpenAI(model=model, temperature=0)
results[model] = run_llm(chat, prompt, model)


print("Huggingfacehub models...")
repo_ids = [
    "google/flan-t5-xxl",
    "databricks/dolly-v2-3b",
    # "Salesforce/xgen-7b-8k-base",
    # "tiiuae/falcon-40b"
]

for model in repo_ids:
    llm = HuggingFaceHub(repo_id=model, model_kwargs={"temperature": 0.1, "max_length": 64})
    results[model] = run_llm(llm, prompt, model)

print("GPT4All...")
gpt4_models = [
    "orca-mini-3b.ggmlv3.q4_0.bin",
    "orca-mini-7b.ggmlv3.q4_0.bin",
    "ggml-mpt-7b-instruct.bin",
    "orca-mini-13b.ggmlv3.q4_0.bin",
    "GPT4All-13B-snoozy.ggmlv3.q4_0.bin",
    "nous-hermes-13b.ggmlv3.q4_0.bin"
]

for model in gpt4_models:
    local_path = f"./models/{model}"
    llm = GPT4All(model=local_path)
    results[model] = run_llm(llm, prompt, model)

print("Results of comparison...")
print(json.dumps(results, indent=2))