#!/usr/bin/env python
# coding: utf-8

# ### Extract Structured Data From Text: Expert Mode (Using Function Calling)
# 
# We are going to explore [OpenAI's Function Calling](https://openai.com/blog/function-calling-and-other-api-updates) for extracting structured data from unstructured sources.
# 
# **Why is this important?**
# LLMs are great at text output, but they need extra help outputing information in a structure that we want. A common request from developers is to get JSON data back from our LLMs.
# 
# Spoiler: Jump down to the bottom to see a bonefied business idea that you can start and manage today.

# In[1]:


# LangChain Models
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Standard Helpers
import pandas as pd
import requests
import time
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Text Helpers
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# For token counting
from langchain.callbacks import get_openai_callback

def printOutput(output):
    print(json.dumps(output,sort_keys=True, indent=3))


# In[2]:


# It's better to do this an environment variable but putting it in plain text for clarity
openai_api_key = os.getenv("OPENAI_API_KEY", 'YourAPIKey')


# Let's start off by creating our LLM. We're using gpt4 to take advantage of its increased ability to follow instructions

# In[3]:


chat = ChatOpenAI(
    model_name="gpt-3.5-turbo-0613", # Cheaper but less reliable
    temperature=0,
    max_tokens=2000,
    openai_api_key=openai_api_key
)


# ### Function Calling Hello World Example
# 
# Create an object that holds information about the fields you'd like to extract

# In[4]:


functions = [
    {
        "name": "get_food_mentioned",
        "description": "Get the food that is mentioned in the review from the customer",
        "parameters": {
            "type": "object",
            "properties": {
                "food": {
                    "type": "string",
                    "description": "The type of food mentioned, ex: Ice cream"
                },
                "good_or_bad": {
                    "type": "string",
                    "description": "whether or not the user thought the food was good or bad",
                    "enum": ["good", "bad"]
                }
            },
            "required": ["location"]
        }
    }
]


# In[5]:


output = chat(messages=
     [
         SystemMessage(content="You are an helpful AI bot"),
         HumanMessage(content="I thought the burgers were awesome")
     ],
     functions=functions
)


# In[6]:


print(json.dumps(output.additional_kwargs, indent=4))


# ### Pydantic Model
# 
# Now let's do the same thing but with a pydantic model rather than json schema

# In[7]:


from langchain.pydantic_v1 import BaseModel, Field
import enum

class GoodOrBad(str, enum.Enum):
    GOOD = "Good"
    BAD = "Bad"

class Food(BaseModel):
    """Identifying information about a person's food review."""

    name: str = Field(..., description="Name of the food mentioned")
    good_or_bad: GoodOrBad = Field(..., description="Whether or not the user thought the food was good or bad")


# In[8]:


output = chat(messages=
     [
         SystemMessage(content="You are an helpful AI bot"),
         HumanMessage(content="I thought the burgers were awesome")
     ],
     functions=[{
         "name": "FoodExtractor",
         "description": (
             "Identifying information about a person's food review."
         ),
         "parameters": Food.schema(),
        }
     ]
)


# In[9]:


output


# But LangChain has an abstraction for us that we can use

# In[10]:


from langchain.chains import create_extraction_chain_pydantic

# Extraction
chain = create_extraction_chain_pydantic(pydantic_schema=Food, llm=chat)

# Run 
text = """I like burgers they are great"""
chain.run(text)


# ### Multiple Results
# 
# Let's try to extract multiple objects from the same text. I'll create a person object now

# In[11]:


from typing import Sequence

chat = ChatOpenAI(
    model_name="gpt-4-0613", # Cheaper but less reliable
    temperature=0,
    max_tokens=2000,
    openai_api_key=openai_api_key
)

class Person(BaseModel):
    """Someone who gives their review on different foods"""

    name: str = Field(..., description="Name of the person")
    foods: Sequence[Food] = Field(..., description="A food that a person mentioned")


# In[12]:


# Extraction
chain = create_extraction_chain_pydantic(pydantic_schema=Person, llm=chat)

# Run 
text = """amy likes burgers and fries but doesn't like salads"""
output = chain.run(text)


# In[13]:


output[0]


# **User Query Extraction**
# 
# Let's do another fun example where we want to extract/convert a query from a user

# In[14]:


class Query(BaseModel):
    """Extract the change a user would like to make to a financial forecast"""

    entity: str = Field(..., description="Name of the category or account a person would like to change")
    amount: int = Field(..., description="Amount they would like to change it by")
    year: int = Field(..., description="The year they would like the change to")


# In[15]:


chain = create_extraction_chain_pydantic(pydantic_schema=Query, llm=chat)


# In[16]:


chain.run("Can you please add 10 more units to inventory in 2022?")


# In[17]:


chain.run("Remove 3 million from revenue in 2021")


# ## Opening Attributes - Real World Example
# 
# [Opening Attributes](https://twitter.com/GregKamradt/status/1643027796850253824) (my sample project for this application)
# 
# If anyone wants to strategize on this project DM me on twitter

# We are going to be pulling jobs from Greenhouse. No API key is needed.

# In[18]:


def pull_from_greenhouse(board_token):
    # If doing this in production, make sure you do retries and backoffs
    
    # Get your URL ready to accept a parameter
    url = f'https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true'
    
    try:
        response = requests.get(url)
    except:
        # In case it doesn't work
        print ("Whoops, error")
        return
        
    status_code = response.status_code
    
    jobs = response.json()['jobs']
    
    print (f"{board_token}: {status_code}, Found {len(jobs)} jobs")
    
    return jobs


# Let's try it out for [Okta](https://www.okta.com/)

# In[19]:


jobs = pull_from_greenhouse("okta")


# Let's look at a sample job with it's raw dictionary

# In[20]:


# Keep in mind that my job_ids will likely change when you run this depending on the postings of the company
job_index = 0


# In[21]:


print ("Preview:\n")
print (json.dumps(jobs[job_index])[:400])


# Let's clean this up a bit

# In[22]:


# I parsed through an output to create the function below
def describeJob(job_description):
    print(f"Job ID: {job_description['id']}")
    print(f"Link: {job_description['absolute_url']}")
    print(f"Updated At: {datetime.fromisoformat(job_description['updated_at']).strftime('%B %-d, %Y')}")
    print(f"Title: {job_description['title']}\n")
    print(f"Content:\n{job_description['content'][:550]}")


# We'll look at another job. This job_id may or may not work for you depending on if the position is still active.

# In[23]:


# Note: I'm using a hard coded job id below. You'll need to switch this if this job ever changes
# and it most definitely will!
job_id = 5299914

job_description = [item for item in jobs if item['id'] == job_id][0]
    
describeJob(job_description)


# I want to convert the html to text, we'll use BeautifulSoup to do this. There are multiple methods you could choose from. Pick what's best for you.

# In[24]:


soup = BeautifulSoup(job_description['content'], 'html.parser')


# In[25]:


text = soup.get_text()

# Convert your html to markdown. This reduces tokens and noise
text = md(text)

print (text[:600])


# Let's create a Kor object that will look for tools. This is the meat and potatoes of the application

# In[26]:


class Tool(BaseModel):
    """The name of a tool or company"""

    name: str = Field(..., description="Name of the food mentioned")
        
class Tools(BaseModel):
    """A tool, application, or other company that is listed in a job description."""

    tools: Sequence[Tool] = Field(..., description=""" A tool or technology listed
        Examples:
        * "Experience in working with Netsuite, or Looker a plus." > NetSuite, Looker
        * "Experience with Microsoft Excel" > Microsoft Excel
    """)


# In[27]:


chain = create_extraction_chain_pydantic(pydantic_schema=Tools, llm=chat)


# In[28]:


output = chain(text)


# In[29]:


output['text']


# [OpenAI GPT4 Pricing](https://help.openai.com/en/articles/7127956-how-much-does-gpt-4-cost)

# In[30]:


with get_openai_callback() as cb:
    result = chain(text)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Successful Requests: {cb.successful_requests}")
    print(f"Total Cost (USD): ${cb.total_cost}")


# Suggested To Do if you want to build this out:
# 
# * Reduce amount of HTML and low-signal text that gets put into the prompt
# * Gather list of 1000s of companies
# * Run through most jobs (You'll likely start to see duplicate information after the first 10-15 jobs per department)
# * Store results
# * Snapshot daily as you look for new jobs
# * Follow [Greg](https://twitter.com/GregKamradt) on Twitter for more tools or if you want to chat about this project
# * Read the user feedback below for what else to build out with this project (I reached out to everyone who signed up on twitter)
# 
# 
# ### Business idea: Job Data As A Service
# 
# Start a data service that collects information about company's jobs. This can be sold to investors looking for an edge.
# 
# After posting [this tweet](https://twitter.com/GregKamradt/status/1643027796850253824) there were 80 people that signed up for the trial. I emailed all of them and most were job seekers looking for companies that used the tech they specialized in.
# 
# The more interesting use case were sales teams + investors.
# 
# #### Interesting User Feedback (Persona: Investor):
# 
# > Hey Gregory, thanks for reaching out. <br><br>
# I always thought that job posts were a gold mine of information, and often suggest identifying targets based on these (go look at relevant job posts for companies that might want to work with you). Secondly, I also automatically ping BuiltWith from our CRM and send that to OpenAI and have a summarized tech stack created - so I see the benefit of having this as an investor. <br><br>
# For me personally, I like to get as much data as possible about a company. Would love to see job post cadence, type of jobs they post and when, notable keywords/phrases used, tech stack (which you have), and any other information we can glean from the job posts (sometimes they have the title of who you'll report to, etc.). <br><br>
# For sales people, I think finer searches, maybe even in natural language if possible - such as "search for companies who posted a data science related job for the first time" - would be powerful.
# 
# If you do this, let me know! I'd love to hear how it goes.
