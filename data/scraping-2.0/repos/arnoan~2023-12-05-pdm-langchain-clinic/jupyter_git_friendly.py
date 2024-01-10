#!/usr/bin/env python
# coding: utf-8

# ### Required Bootstrapping: Setup of Credentials
# Make sure to populate the `.env` file with your OpenAI API key.  
# See `.env.template` for an example.  

# In[ ]:


import dotenv


# In[ ]:


dotenv.load_dotenv()


# In[ ]:


import os


# In[ ]:


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# What do we need to build the most simple LLM system possible?
# 
# - a model
# - a question / a prompt

# In[ ]:


from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.schema.output_parser import StrOutputParser


# In[ ]:


model = ChatOpenAI(
  model="gpt-4",
  openai_api_key=OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

output_parser = StrOutputParser()


# In[ ]:


chain = prompt | model | output_parser


# In[ ]:


prompt


# In[ ]:


prompt.invoke({"topic": "cats"})


# In[ ]:


chain.invoke({"topic": "cats"})


# ## Input & Output Schema

# In[ ]:


chain.input_schema.schema_json()


# In[ ]:


chain.output_schema.schema_json()


# In[ ]:


from pprint import pprint


# In[ ]:


pprint((prompt | model).output_schema.schema_json())


# In[ ]:




