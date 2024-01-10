# -*- coding: utf-8 -*-
"""
https://www.langchain.com.cn/modules/chains/getting_started
https://python.langchain.com/docs/modules/chains/
"""
import os
import torch as th
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
                          T5Tokenizer, T5ForConditionalGeneration, pipeline)
from langchain.llms import OpenAI
from langchain import (HuggingFaceHub, PromptTemplate, LLMChain)
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 快速开始
llm = OpenAI(temperature=0.9)
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("colorful socks"))

chat = ChatOpenAI(temperature=0.9)
hum_template = "What is a good name for a company that makes {product}?"
hum_prompt = HumanMessagePromptTemplate.from_template(hum_template)
chat_prompt = ChatPromptTemplate.from_messages([hum_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
print(chain.run("colorful socks"))
print(chain.run("colorful socks"))

# ----------------------------------------------------------------------------------------------------------------
# 向链中添加内存
conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory(),
    verbose=True
)

conversation.run("Answer briefly. What are the first 3 colors of a rainbow?")
# -> The first three colors of a rainbow are red, orange, and yellow.
conversation.run("And the next 4?")
# -> The next four colors of a rainbow are green, blue, indigo, and violet.
 

