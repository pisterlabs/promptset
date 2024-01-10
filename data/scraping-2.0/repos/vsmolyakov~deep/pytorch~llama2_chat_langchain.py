from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

import transformers
import torch
import accelerate
import warnings
warnings.filterwarnings('ignore')

model="meta-llama/Llama-2-7b-chat-hf"
tokenizer=AutoTokenizer.from_pretrained(model)
pipeline=transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
    )

llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.7})

prompt_template = """<s>[INST] <<SYS>>
{{ You are a helpful AI Assistant}}<<SYS>>
###

Previous Conversation:
'''
{history}
'''

{{{input}}}[/INST]

"""
prompt = PromptTemplate(template=prompt_template, input_variables=['input', 'history'])
memory = ConversationBufferWindowMemory(k=5)

chain = ConversationChain(llm=llm, prompt=prompt, memory=memory)
chain.run("What are different ways of generating income?")