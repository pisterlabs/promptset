import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from finetune_utils import load_finetune_config
from finetune_lit import load_peft_model
from langchain_lit import load_markdown_documents, LlmEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough

from pdf_utils import load_pdf_documents_from_directory

config = load_finetune_config()
device = "cuda"
EMB_MODEL = "bge-base-en"


def load_vector_store():
    print("loading data")
    md_docs = load_markdown_documents("./data")
    pdf_docs = load_pdf_documents_from_directory('./data')
    docs = pdf_docs + md_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=35)
    all_splits = text_splitter.split_documents(docs)
    llm_embedding = LlmEmbedding(f"../models/{EMB_MODEL}")
    print("loading vector")
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=llm_embedding.embedding)
    return vectorstore


def clean_prompt_resp(resp: str):
    after_inst = resp.split("[/INST]", 1)[-1]
    s2 = after_inst.split("[INST]", 1)[0]
    return s2.split('[/INST]', 1)[0]


model_name = config['model_name']
base_model = f"../models/{model_name}"
peft_model_id = f"./outputs/{model_name}-qlora"

model, tokenizer = load_peft_model(base_model, peft_model_id)

generation_config = model.generation_config
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.01  # 0.7
generation_config.top_p = 0.9
generation_config.do_sample = True
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id


SYSTEM_PROMPT = """"""
prompt_template = """<s>[INST] {user_input} [/INST]"""

generation_pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    # max_length=4096,
    temperature=0.01,
    top_p=2,
    repetition_penalty=1.15,
    return_full_text=True,
)

prompt_template = """
### [INST] 
Instruction: Answer the question based on your gaming knowledge. 
If the answer cannot be found from the context, try to find the answer from your knowledge. 
If still unable to find the answer, respond with 'I don't know.'.
Here is context to help:

{context}

### QUESTION:
{question} 

[/INST]
"""


prompt_template = """
### [INST] 
Instruction: Answer the question based on your gaming knowledge. 
If the answer cannot be found from the context, respond with 'I don't know.'.
Here is context to help:

{context}

### QUESTION:
{question} 

[/INST]
"""

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm = HuggingFacePipeline(pipeline=generation_pipe)
llm_chain = LLMChain(llm=llm, prompt=prompt)

print("load db")
vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(
    search_kwargs={'k': 10, 'fetch_k': 50}
    )

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | llm_chain
)


def ask_qa(user_input):
    resp = rag_chain.invoke(user_input)
    doc = resp['context'][0]
    page_content = doc.page_content
    source = doc.metadata['source']
    answer = resp['text']
    print(f"{source=}")
    return answer


def ask_llm(user_input):
   prompt_template2 = """
[INST]    
{context}
[/INST]
"""
   prompt = prompt_template2.format(context=user_input)
   encoding = tokenizer(prompt, return_tensors="pt").to(device)

   outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config
   )

   resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
   answer = resp.replace(prompt, "")
   # answer = answer.strip().replace("### ANSWER:\n", "")
   return answer


print(f"load {model_name} done")
while True:
    user_input = input("query: ")
    if user_input == '/bye':
        break
    if user_input == '':
        continue

    answer = ask_qa(user_input)
    print("--------------------------------------------------")
    print(answer)
    print("")
    print("--------------------------------------------------")
