#!/usr/bin/env python
# coding: utf-8

# ## Quickstart Guide
# https://langchain.readthedocs.io/en/latest/getting_started/getting_started.html

# In[1]:


# pip install langchain openai google-search-results



# In[9]:


import os
# os.environ["OPENAI_API_KEY"] = ""


# # Building A Language Model Application
# ### LLMS: Get predictions from a language model

# In[10]:


from langchain.llms import OpenAI


# In[11]:


llm = OpenAI(temperature=0.9)


# In[12]:


text = "What are 5 vacation destinations for someone who likes to eat pasta?"
print(llm(text))


# ### Prompt Templates: Manage prompts for LLMs

# In[13]:


from langchain.prompts import PromptTemplate


# In[14]:


prompt = PromptTemplate(
    input_variables=["food"],
    template="What are 5 vacation destinations for someone who likes to eat {food}?",
)


# In[15]:


print(prompt.format(food="dessert"))


# In[16]:


print(llm(prompt.format(food="dessert")))


# ### Chains: Combine LLMs and prompts in multi-step workflows

# In[17]:


from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


# In[18]:


llm = OpenAI(temperature=0.9)

prompt = PromptTemplate(
    input_variables=["food"],
    template="What are 5 vacation destinations for someone who likes to eat {food}?",
)


# In[19]:


chain = LLMChain(llm=llm, prompt=prompt)


# In[20]:


print(chain.run("fruit"))


# ### Agents: Dynamically call chains based on user input

# In[ ]:





# In[21]:


from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI


# In[22]:


# Load the model
llm = OpenAI(temperature=0)


# In[23]:


# Load in some tools to use

# os.environ["SERPAPI_API_KEY"] = "..."

tools = load_tools(["serpapi", "llm-math"], llm=llm)


# In[24]:


# Finally, let's initialize an agent with:
# 1. The tools
# 2. The language model
# 3. The type of agent we want to use.

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


# See list of agents types [here](https://python.langchain.com/docs/modules/agents/agent_types/)

# In[25]:


# Now let's test it out!
agent.run("Who is the current leader of Japan? What is the largest prime number that is smaller than their age?")


# ### Memory: Add state to chains and agents

# In[26]:


from langchain import OpenAI, ConversationChain


# In[27]:


llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)


# In[28]:


conversation.predict(input="Hi there!")


# In[29]:


conversation.predict(input="I'm doing well! Just having a conversation with an AI.")


# In[30]:


conversation.predict(input="What was the first thing I said to you?")


# In[31]:


conversation.predict(input="what is an alternative phrase for the first thing I said to you?")


# In[ ]:




