#!/usr/bin/env python
# coding: utf-8

# ### Extract Structured Data From Text: Expert Mode (Using Kor)
# 
# For complicated data extraction you need a robust library. The [Kor Library](https://eyurtsev.github.io/kor/nested_objects.html) (created by [Eugene Yurtsev](https://twitter.com/veryboldbagel)) is an awesome tool just for this.
# 
# We are going to explore using Kor with a practical use case.
# 
# **Why is this important?**
# LLMs are great at text output, but they need extra help outputing information in a structure that we want. A common request from developers is to get JSON data back from our LLMs.
# 
# Spoiler: Jump down to the bottom to see a bonefied business idea that you can start and manage today.

# In[1]:


# Kor!
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# LangChain Models
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Standard Helpers
import pandas as pd
import requests
import time
import json
from datetime import datetime

# Text Helpers
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# For token counting
from langchain.callbacks import get_openai_callback

def printOutput(output):
    print(json.dumps(output,sort_keys=True, indent=3))


# In[2]:


# It's better to do this an environment variable but putting it in plain text for clarity
openai_api_key = 'your_api_key'


# In[34]:


openai_api_key = '...'


# Let's start off by creating our LLM. We're using gpt4 to take advantage of its increased ability to follow instructions

# In[4]:


llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo", # Cheaper but less reliable
    model_name="gpt-4",
    temperature=0,
    max_tokens=2000,
    openai_api_key=openai_api_key
)


# ### Kor Hello World Example
# 
# Create an object that holds information about the fields you'd like to extract

# In[5]:


person_schema = Object(
    # This what will appear in your output. It's what the fields below will be nested under.
    # It should be the parent of the fields below. Usually it's singular (not plural)
    id="person",
    
    # Natural language description about your object
    description="Personal information about a person",
    
    # Fields you'd like to capture from a piece of text about your object.
    attributes=[
        Text(
            id="first_name",
            description="The first name of a person.",
        )
    ],
    
    # Examples help go a long way with telling the LLM what you need
    examples=[
        ("Alice and Bob are friends", [{"first_name": "Alice"}, {"first_name": "Bob"}])
    ]
)


# Create a chain that will extract the information and then parse it. This uses LangChain under the hood

# In[6]:


chain = create_extraction_chain(llm, person_schema)


# In[7]:


text = """
    My name is Bobby.
    My sister's name is Rachel.
    My brother's name Joe. My dog's name is Spot
"""
output = chain.predict_and_parse(text=(text))["data"]

printOutput(output)
# Notice how there isn't "spot" in the results list because it's the name of a dog, not a person.


# Kor also facilitates returning None when the LLM doesn't find what you're looking for

# In[8]:


output = chain.predict_and_parse(text=("The dog went to the park"))["data"]
printOutput(output)


# ### Multiple Fields
# You can pass multiple fields if you're looking for more information

# In[9]:


plant_schema = Object(
    id="plant",
    description="Information about a plant",
    
    # Notice I put multiple fields to pull out different attributes
    attributes=[
        Text(
            id="plant_type",
            description="The common name of the plant."
        ),
        Text(
            id="color",
            description="The color of the plant"
        ),
        Number(
            id="rating",
            description="The rating of the plant."
        )
    ],
    examples=[
        (
            "Roses are red, lilies are white and a 8 out of 10.",
            [
                {"plant_type": "Roses", "color": "red"},
                {"plant_type": "Lily", "color": "white", "rating" : 8},
            ],
        )
    ]
)


# In[10]:


text="Palm trees are brown with a 6 rating. Sequoia trees are green"

chain = create_extraction_chain(llm, plant_schema)
output = chain.predict_and_parse(text=text)['data']

printOutput(output)


# ### Working With Lists
# 
# You can also extract lists as well.
# 
# Note: Check out how I have a nested object. The 'parts' object is in the 'cars_schema'

# In[11]:


parts = Object(
    id="parts",
    description="A single part of a car",
    attributes=[
        Text(id="part", description="The name of the part")
    ],
    examples=[
        (
            "the jeep has wheels and windows",
            [
                {"part": "wheel"},
                {"part": "window"}
            ],
        )
    ]
)

cars_schema = Object(
    id="car",
    description="Information about a car",
    examples=[
        (
            "the bmw is red and has an engine and steering wheel",
            [
                {"type": "BMW", "color": "red", "parts" : ["engine", "steering wheel"]}
            ],
        )
    ],
    attributes=[
        Text(
            id="type",
            description="The make or brand of the car"
        ),
        Text(
            id="color",
            description="The color of the car"
        ),
        parts
    ]
)


# In[12]:


# To do nested objects you need to specify encoder_or_encoder_class="json"
text = "The blue jeep has rear view mirror, roof, windshield"

# Changed the encoder to json
chain = create_extraction_chain(llm, cars_schema, encoder_or_encoder_class="json")
output = chain.predict_and_parse(text=text)['data']

printOutput(output)


# View the prompt that was sent over

# In[13]:


prompt = chain.prompt.format_prompt(text=text).to_string()

print(prompt)


# Kor is a really great way to extract actions from a user as well

# In[14]:


schema = Object(
  id="forecaster",
  description=(
      "User is controling an app that makes financial forecasts. "
      "They will give a command to update a forecast in the future"
  ),
  attributes=[
      Text(
          id="year",
          description="Year the user wants to update",
          examples=[("please increase 2014's customers by 15%", "2014")],
          many=True,
      ),
      Text(
          id="metric",
          description="The unit or metric a user would like to influence",
          examples=[("please increase 2014's customers by 15%", "customers")],
          many=True,
      ),
      Text(
          id="amount",
          description="The quantity of a forecast adjustment",
          examples=[("please increase 2014's customers by 15%", ".15")],
          many=True,
      )
    ],
  many=False,
)


# In[15]:


chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
output = chain.predict_and_parse(text="please add 15 more units sold to 2023")['data']

printOutput(output)


# ## Opening Attributes - Real World Example
# 
# [Opening Attributes](https://twitter.com/GregKamradt/status/1643027796850253824) (my sample project for this application)
# 
# If anyone wants to strategize on this project DM me on twitter

# In[16]:


llm = ChatOpenAI(
    # model_name="gpt-3.5-turbo",
    model_name="gpt-4",
    temperature=0,
    max_tokens=2000,
    openai_api_key=openai_api_key
)


# We are going to be pulling jobs from Greenhouse. No API key is needed.

# In[17]:


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

# In[18]:


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
job_id = 4982726

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


tools = Object(
    id="tools",
    description="""
        A tool, application, or other company that is listed in a job description.
        Analytics, eCommerce and GTM are not tools
    """,
    attributes=[
        Text(
            id="tool",
            description="The name of a tool or company"
        )
    ],
    examples=[
        (
            "Experience in working with Netsuite, or Looker a plus.",
            [
                {"tool": "Netsuite"},
                {"tool": "Looker"},
            ],
        ),
        (
           "Experience with Microsoft Excel",
            [
               {"tool": "Microsoft Excel"}
            ] 
        ),
        (
           "You must know AWS to do well in the job",
            [
               {"tool": "AWS"}
            ] 
        ),
        (
           "Troubleshooting customer issues and debugging from logs (Splunk, Syslogs, etc.) ",
            [
               {"tool": "Splunk"},
            ] 
        )
    ],
    many=True,
)


# In[27]:


chain = create_extraction_chain(llm, tools, input_formatter="triple_quotes")


# In[28]:


output = chain.predict_and_parse(text=text)["data"]

printOutput(output)


# ### Salary
# 
# Let's grab salary information while we are at it.
# 
# Not all jobs will list this information. If they do, it's rarely consistent across jobs. A great use case for LLMs to catch this information!

# In[29]:


salary_range = Object(
    id="salary_range",
    description="""
        The range of salary offered for a job mentioned in a job description
    """,
    attributes=[
        Number(
            id="low_end",
            description="The low end of a salary range"
        ),
        Number(
            id="high_end",
            description="The high end of a salary range"
        )
    ],
    examples=[
        (
            "This position will make between $140 thousand and $230,000.00",
            [
                {"low_end": 140000, "high_end": 230000},
            ]
        )
    ]
)


# In[30]:


jobs = pull_from_greenhouse("cruise")


# In[31]:


# This job id may not work for you, pick another one from the list if it doesn't.
job_id = 4858414

job_description = [item for item in jobs if item['id'] == job_id][0]
    
describeJob(job_description)

soup = BeautifulSoup(job_description['content'], 'html.parser')
text = soup.get_text()

# Convert your html to markdown. This reduces tokens and noise
text = md(text)

print (text[:600])


# In[32]:


chain = create_extraction_chain(llm, salary_range)
output = chain.predict_and_parse(text=text)["data"]

printOutput(output)


# > The salary range for this position is $112,300 - 165,000. Compensation will vary depending on location, job-related knowledge, skills, and experience. You may also be offered a bonus, restricted stock units, and benefits. These ranges are subject to change.
# 
# Awesome!

# [OpenAI GPT4 Pricing](https://help.openai.com/en/articles/7127956-how-much-does-gpt-4-cost)

# In[33]:


with get_openai_callback() as cb:
    result = chain.predict_and_parse(text=text)
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
