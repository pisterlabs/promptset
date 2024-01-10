#!/usr/bin/env python
# coding: utf-8

# ### Extract Data From Calls & Video Transcripts/Interviews
# 
# There is a lot of spoke word out there. Podcasts, interviews, sales calls, phone calls.
# 
# It is extremely useful to be able to extract this information and create value with it. We're going to run through an example focusing on the B2B sales use case.
# 
# **Small plug:** This is a mini-preview on how I built [Thimble](https://thimbleai.com/) which helps Account Executives pull information from their sales calls. Sign up at https://thimbleai.com/ if you want to check it out or here's the [demo video](https://youtu.be/DIw4rbpI9ic) or see more information on [Twitter](https://twitter.com/GregKamradt)
# 
# Plug over, Let's learn! 😈
# 
# Through building Thimble I've learned a few tricks to make working with transcripts easier:
# 1. **Name/Role -** Put the name of each person before their sentence. Bonus points if you have their role/company included too. Example: Greg (Marin Transitions): Hey! How's it going?
# 2. **System instructions -** Be specific with your system prompt about the role you need you bot to play
# 3. **Only pull from the call -** Emphasize not to make anything up
# 4. **Don't make the user prompt -** Abstract away any user prompting necessary with key:value pairs
# 
# First let's import our packages

# In[1]:


# To get environment variables
import os

# Make the display a bit wider
from IPython.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

# To split our transcript into pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Our chat model. We'll use the default which is gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

# Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, # I included this one so you know you'll have it but we won't be using it
    HumanMessagePromptTemplate
)

# To create our chat messages
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


# In[22]:


os.environ['OPENAI_API_KEY'] = '...'


# Then let's load up our data. This is already a formatted transcript of a mock sales call. 70% of the hard part is getting clean data. Don't under estimate the reward you get for cleaning up your data first!

# In[3]:


with open('../data/Transcripts/acme_co_v2.txt', 'r') as file:
    content = file.read()


# In[4]:


print ("Transcript:\n")
print(content[:215]) # Why 215? Because it cut off at a clean line


# Split our documents so we don't run into token issues. Experiment with what chunk size words best for your use case

# In[5]:


text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
texts = text_splitter.create_documents([content])


# In[6]:


print (f"You have {len(texts)} texts")
texts[0]


# In[7]:


# Your api key should be an environment variable, or else put it here
# We are using a chat model in case you wanted to use gpt4
llm = ChatOpenAI(temperature=0)


# We're going to start with the vanilla load_summarize_chain to see how it goes.
# If you want to see the default prompts that are used you can explore the LangChain code. Here are the [map reduce prompts](https://github.com/hwchase17/langchain/blob/master/langchain/chains/summarize/map_reduce_prompt.py)

# In[8]:


# verbose=True will output the prompts being sent to the 
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)


# **Note:** At this point you might be asking, "Greg, what about embeddings?" Since we're just doing one call, I'm not worry about embeddings right now and putting the whole thing in a map reduce summarizer. If you have multiple calls and a lot of history then you should be embedding them for retrieval later. See tutorial on embeddings [here.](https://youtu.be/h0DHDp1FbmQ). [Tutorial on chain types.](https://youtu.be/f9_BWhCI4Zo)

# In[9]:


output = chain.run(texts)


# In[10]:


print (output)


# Not bad, but it's giving me the perspective of a 3rd party watching the conversation that is agnostic to the content. I want the AI bot to me on my side! I'll need to switch up the prompts to do this.

# ### Custom Prompts
# 
# I'm going to write custom prompts that give the AI more instructions on what role I want it to play

# In[11]:


template="""

You are a helpful assistant that helps {sales_rep_name}, a sales rep at {sales_rep_company}, summarize information from a sales call.
Your goal is to write a summary from the perspective of {sales_rep_name} that will highlight key points that will be relevant to making a sale
Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"
Do not repeat {sales_rep_name}'s name in your output

"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template="{text}" # Simply just pass the text as a human message
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


# In[12]:


chat_prompt = ChatPromptTemplate.from_messages(messages=[system_message_prompt, human_message_prompt])


# In[13]:


chain = load_summarize_chain(llm,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt
                            )

# Because we aren't specifying a combine prompt the default one will be used


# In[14]:


output = chain.run({
                    "input_documents": texts,
                    "sales_rep_company": "Marin Transitions Partner", \
                    "sales_rep_name" : "Greg"
                   })


# In[15]:


print (output)


# Better! But say I wanted to change the format of the output without needing the user to do extra prompting.

# ### Promptless changes
# 
# To do this I'll write a few points about the different output types I would like. However, I'll that I'll expose to the user is a simple selection, radio button, or drop down. (We'll use text for now but you can do this in your app).
# 
# I want to give the user the option to select between different summary output types.
# 
# I'll have them pick between:
# 1. One Sentence
# 2. Bullet Points
# 3. Short
# 4. Long
# 
# I could try to pass these words to the LLM, but I want to be more explicit with it. Plus, giving good instructions is the way to go!

# In[16]:


summary_output_options = {
    'one_sentence' : """
     - Only one sentence
    """,
    
    'bullet_points': """
     - Bullet point format
     - Separate each bullet point with a new line
     - Each bullet point should be concise
    """,
    
    'short' : """
     - A few short sentences
     - Do not go longer than 4-5 sentences
    """,
    
    'long' : """
     - A verbose summary
     - You may do a few paragraphs to describe the transcript if needed
    """
}


# Create a new template that takes an additional parameter. I need to put this in the combined prompt so that the LLM will output in my format. If I did this in the map section I would lose the format after the combined prompt was done
# 
# **Map Prompt**

# In[17]:


template="""

You are a helpful assistant that helps {sales_rep_name}, a sales rep at {sales_rep_company}, summarize information from a sales call.
Your goal is to write a summary from the perspective of Greg that will highlight key points that will be relevant to making a sale
Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"
"""
system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

human_template="{text}" # Simply just pass the text as a human message
human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])


# **Combined Prompt**

# In[18]:


template="""

You are a helpful assistant that helps {sales_rep_name}, a sales rep at {sales_rep_company}, summarize information from a sales call.
Your goal is to write a summary from the perspective of Greg that will highlight key points that will be relevant to making a sale
Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"

Respond with the following format
{output_format}

"""
system_message_prompt_combine = SystemMessagePromptTemplate.from_template(template)

human_template="{text}" # Simply just pass the text as a human message
human_message_prompt_combine = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_combine, human_message_prompt_combine])


# In[19]:


chain = load_summarize_chain(llm,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt_map,
                             combine_prompt=chat_prompt_combine,
                             verbose=True
                            )


# In[20]:


user_selection = 'one_sentence'

output = chain.run({
                    "input_documents": texts,
                    "sales_rep_company": "Marin Transitions Partner", \
                    "sales_rep_name" : "Greg",
                    "output_format" : summary_output_options[user_selection]
                   })


# In[21]:


print(output)


# Awesome! Now we have a bullet point format without needing to have the user specify any additional information.
# 
# If you wanted to productionize this you would need to add additional prompts to extract other information from the calls that may be helpful to a sales person. Example: Key Points + Next Steps from the call. You should also parallelize the map calls if you do the map reduce method.
# 
# Have other ideas about how something like this could be used? Send me a tweet or DM on [Twitter](https://twitter.com/GregKamradt) or contact@dataindependent.com
