import asyncio

import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import textwrap
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate

DB_FAISS_PATH = "models/db_faiss"
MODEL_PTH_NAME = "llama-2-7b-chat.ggmlv3.q8_0.bin"

MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
print("tokenizer")

# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
#                                              device_map='auto',
#                                              torch_dtype=torch.float16,
#                                              use_auth_token=True,
#                                              # load_in_4bit=True
#                                              )
print("model")


MODEL_NAME = 'models/llama-2-7b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             # load_in_4bit=True
                                             )
# 可以匯出 .cache 模型到本地
# model.save_pretrained("models/llama-2-7b-chat-hf")


pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n</SYS>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible.
If a question does not make any sense, or is not factually coherent, 
explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information.
"""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
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


def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)
    return final_outputs  # , outputs


def parse_text(text):
    wrapped_text = textwrap.fill(text, width=100)
    print(wrapped_text + '\n\n')


llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})


def demo1():
    instruction = "What is the temperature in Melbourne?"
    template = get_prompt(instruction)
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    text = "how are you today?"
    output = llm_chain.run(text)


def demo_chat():
    instruction = "Chat History:\n\n{chat_history}\n\nUser: {user_input}"
    system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"
    template = get_prompt(instruction, system_prompt)
    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"],
        template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    while True:
        user_input = input("query: ")
        if user_input == 'q':
            break
        answer = llm_chain.predict(user_input=user_input)
        print(answer)


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

from langchain.embeddings import LlamaCppEmbeddings
#llama = LlamaCppEmbeddings(model_path=MODEL_NAME)

def demo_txt():
    instruction = "Chat History:\n\n{chat_history}\n\nUser: {user_input}"
    system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"
    template = get_prompt(instruction, system_prompt)
    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"],
        template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    while True:
        user_input = input("query: ")
        if user_input == 'q':
            break
        answer = llm_chain.predict(user_input=user_input)
        print(answer)
