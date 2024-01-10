from utils import llm
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import LLMChain
from prompts import *



# QA ConversationChain


summarize_chain = LLMChain(llm=llm, prompt=summary_prompt)
short_summary_chain = LLMChain(llm=llm, prompt=short_summary_prompt)
title_chain = LLMChain(llm=llm, prompt=title_prompt)
formatting_chain = LLMChain(llm=llm, prompt=formatting_prompt)