#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling With Language Models
# 
# *View this code on [Github](https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/Topic%20Modeling%20With%20Language%20Models.ipynb)*
# 
# 
# Topic Modeling is the practice of pulling out categorized groups of information from a piece of longer text.
# 
# Example: Inferring chapters from a book or segments of a movie. This is a classic data science topic that has been studied for years. Check out examples from previous research like [scikit-learn](https://towardsdatascience.com/introduction-to-topic-modeling-using-scikit-learn-4c3f3290f5b9) and [BERTopic](https://www.youtube.com/watch?v=uZxQz87lb84) if you want to see these techniques.
# 
# However today we are going to take a pass at this problem using language models. Why? Language models are extremely good at processing text and pulling out big picture ideas from a document.
# 
# There are many methods to do this and my goal for today's tutorial is to show you a few different approaches so you can apply it to your own scenario.
# 
# In this lesson we are prioritizing comprehensiveness and robustness of information over API costs so please be mindful of your expense comfortability.
# 
# I'll be taking a 2-pass approach today:
# * **1st Pass:** Run through the entire document via map reduce and pull out topics as bullet points
# * **2nd Pass:** Iterate through your topic bullet points and expand on them with a subset of context that was selected via retrieval
# 
# 
# Today we are going to be looking at a [My First Million](https://www.mfmpod.com/) podcast because it's rich with segments, ideas, sayings, and stories. Great for topic parsing!
# 
# **Bonus**: As a bonus we are also going to be looking at how to auto generate timestamps for each topic as well. The most common use case of this is YouTube Chapters
# 
# **My Assumptions**
# * You don't have a table of contents. That would definitely help out (since a human likely generated them) but I want to make this method as general as possible so you can apply it
# * You want to *learn* the nuts and bolts how to do this. If you wanted a 3rd party tool to do this for you I suggest something like [AssemblyAI](https://www.assemblyai.com/) or [PodcastNotes](https://podcastnotes.org/) 
# 
# ### Use Cases:
# 
# * **YouTube Videos** - Auto Chapter Generation
# * **Podcasts** - Extract structured information
# * **Meeting Notes** - Send topic summaries to participants
# * **Town Hall Meetings** - Structured information
# * **Earnings Report Calls** - Sell structured data to investment groups
# * **Legal Documents** - Quickly summarize by topic
# * **Movie Scripts** - Quick bullet points for production recaps
# * **Books** - Auto generate table of contents
# 
# Finally, if you want to see the inspiration for this tutorial, [here's the tweet](https://twitter.com/GregKamradt/status/1651957952725807106) that started it all.
# 
# Let's get started!

# In[20]:


# Make the display a bit wider
# from IPython.display import display, HTML
# display(HTML("<style>.container { width:90% !important; }</style>"))

# LangChain basics
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain

# Vector Store and retrievals
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma, Pinecone
import pinecone

# Chat Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Supporting libraries
import os
from dotenv import load_dotenv

load_dotenv()


# ### The Set Up - Create your LLMs and get data

# In[2]:


# Creating two versions of the model so I can swap between gpt3.5 and gpt4
llm3 = ChatOpenAI(temperature=0,
                  openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'),
                  model_name="gpt-3.5-turbo-0613",
                  request_timeout = 180
                )

llm4 = ChatOpenAI(temperature=0,
                  openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'),
                  model_name="gpt-4-0613",
                  request_timeout = 180
                 )


# First we'll need to get transcripts. I put a few pre-processed transcripts in the data folder of this repo.
# 
# If you need transcripts for your own audio I suggest a transcription tool like [AssemblyAI](https://www.assemblyai.com/). I also tried [Steno.ai](https://steno.ai/my-first-million) but the quality and speaker detection wasn't that high.
# 
# Reach out if you want me to grab transcripts in bulk for you.

# In[3]:


# I put three prepared transcripts
transcript_paths = [
    '../data/Transcripts/MFMPod/mfm_pod_steph.txt',
    '../data/Transcripts/MFMPod/mfm_pod_alex.txt',
    '../data/Transcripts/MFMPod/mfm_pod_rob.txt'
]

with open('../data/Transcripts/MFMPod/mfm_pod_steph.txt') as file:
    transcript = file.read()


# In[4]:


print(transcript[:280])


# Then we are going to split our text up into chunks. We do this so:
# 
# 1. The context size is smaller and the LLM can increase it's attention to context ratio
# 2. In case the text is too long and it wouldn't fit in the prompt anyway

# In[5]:


# Load up your text splitter
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=10000, chunk_overlap=2200)

# I'm only doing the first 23250 characters. This to save on costs. When you're doing your exercise you can remove this to let all the data through
transcript_subsection_characters = 23250
docs = text_splitter.create_documents([transcript[:transcript_subsection_characters]])
print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")


# ## Step 1: Extract Topic Titles & Short Description
# 
# ### The Custom Prompts - Customize your prompt to fit your use case

# Next up I'm going to use custom prompts to instruct the LLM on how to pull out the topics I want.
# 
# This will be heavily dependent on your domain. You should adjust the prompt below to use descriptions and examples that are relevant to you.
# 
# I built these descriptions over many iterations playing with prompts and checking the output. If you ever start a business this will be part of your IP!
# 
# I will ask the LLM for a topic title and a short description. I found it was too much for the LLM to ask for a long description in the first pass. Results weren't great and high latency.
# 
# Let's start with our map prompt which will iterate over the chunks we just made

# In[6]:


template="""
You are a helpful assistant that helps retrieve topics talked about in a podcast transcript
- Your goal is to extract the topic names and brief 1-sentence description of the topic
- Topics include:
  - Themes
  - Business Ideas
  - Interesting Stories
  - Money making businesses
  - Quick stories about people
  - Mental Frameworks
  - Stories about an industry
  - Analogies mentioned
  - Advice or words of caution
  - Pieces of news or current events
- Provide a brief description of the topics after the topic name. Example: 'Topic: Brief Description'
- Use the same words and terminology that is said in the podcast
- Do not respond with anything outside of the podcast. If you don't see any topics, say, 'No Topics'
- Do not respond with numbers, just bullet points
- Do not include anything about 'Marketing Against the Grain'
- Only pull topics from the transcript. Do not use the examples
- Make your titles descriptive but concise. Example: 'Shaan's Experience at Twitch' should be 'Shaan's Interesting Projects At Twitch'
- A topic should be substantial, more than just a one-off comment

% START OF EXAMPLES
 - Sam’s Elisabeth Murdoch Story: Sam got a call from Elizabeth Murdoch when he had just launched The Hustle. She wanted to generate video content.
 - Shaan’s Rupert Murdoch Story: When Shaan was running Blab he was invited to an event organized by Rupert Murdoch during CES in Las Vegas.
 - Revenge Against The Spam Calls: A couple of businesses focused on protecting consumers: RoboCall, TrueCaller, DoNotPay, FitIt
 - Wildcard CEOs vs. Prudent CEOs: However, Munger likes to surround himself with prudent CEO’s and says he would never hire Musk.
 - Chess Business: Priyav, a college student, expressed his doubts on the MFM Facebook group about his Chess training business, mychesstutor.com, making $12.5K MRR with 90 enrolled.
 - Restaurant Refiller: An MFM Facebook group member commented on how they pay AirMark $1,000/month for toilet paper and toilet cover refills for their restaurant. Shaan sees an opportunity here for anyone wanting to compete against AirMark.
 - Collecting: Shaan shared an idea to build a mobile only marketplace for a collectors’ category; similar to what StockX does for premium sneakers.
% END OF EXAMPLES
"""
system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

human_template="Transcript: {text}" # Simply just pass the text as a human message
human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])


# Then we have our combine prompt which will run once over the results of the map prompt above

# In[7]:


template="""
You are a helpful assistant that helps retrieve topics talked about in a podcast transcript
- You will be given a series of bullet topics of topics vound
- Your goal is to exract the topic names and brief 1-sentence description of the topic
- Deduplicate any bullet points you see
- Only pull topics from the transcript. Do not use the examples

% START OF EXAMPLES
 - Sam’s Elisabeth Murdoch Story: Sam got a call from Elizabeth Murdoch when he had just launched The Hustle. She wanted to generate video content.
 - Shaan’s Rupert Murdoch Story: When Shaan was running Blab he was invited to an event organized by Rupert Murdoch during CES in Las Vegas.
% END OF EXAMPLES
"""
system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

human_template="Transcript: {text}" # Simply just pass the text as a human message
human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])


# ### The First Pass - Run through your text and extract the topics per your custom prompts
# Then we get our chain ready. This is object that will do the actual processing for us when we call it. I'm using gpt4 because we need the increased reasoning ability to pull out topics. You could use gpt3.5 but results may vary.

# In[8]:


chain = load_summarize_chain(llm4,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt_map,
                             combine_prompt=chat_prompt_combine,
#                              verbose=True
                            )


# Then the `.run()` code below will do the actual API calls and work

# In[9]:


topics_found = chain.run({"input_documents": docs})


# In[10]:


print (topics_found)


# ### Structured Data - Turn your LLM output into structured data
# 
# The LLM just returned a wall of text to us, I want to convert this into structured data I can more easily use elsewhere.
# 
# We might have been able to do add structured output instructions to the pull above but I preferred to do it in two steps for clarity. Plus the cost us super low so we only have latency to worry about, but that isn't a priority for this tutorial.
# 
# We will use OpenAI's [function calling](function Calling via ChatGPT API - First Look With LangChain - YouTube) to extract each topic.

# In[11]:


schema = {
    "properties": {
        # The title of the topic
        "topic_name": {
            "type": "string",
            "description" : "The title of the topic listed"
        },
        # The description
        "description": {
            "type": "string",
            "description" : "The description of the topic listed"
        },
        "tag": {
            "type": "string",
            "description" : "The type of content being described",
            "enum" : ['Business Models', 'Life Advice', 'Health & Wellness', 'Stories']
        }
    },
    "required": ["topic", "description"],
}


# In[12]:


# Using gpt3.5 here because this is an easy extraction task and no need to jump to gpt4
chain = create_extraction_chain(schema, llm3)


# In[13]:


topics_structured = chain.run(topics_found)


# In[14]:


topics_structured


# Great, now we have our structured topics. Let's move into the next step and *expand* on those topics even more.
# 
# ## Step 2: Expand on the topics you found
# 
# In order to expand on the topics we found we are going to do the vectorstore dance. We'll chunk our podcast into *small* chunks and then modify the retrieval and qa chain to help us pull out more information.
# 
# I want to split into small chunks to hopefully increase the signal to noise ratio. Here I'll only do 4K characters which is less than half of what we did above.

# In[15]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)

docs = text_splitter.create_documents([transcript[:transcript_subsection_characters]])

print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")


# Because I want to do Question & Answer Retrieval, we need to get embeddings for our documents so we can pull out the docs which are similar for context later.

# In[16]:


embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'))


# ### Option #1: Pinecone
# 
# Use this if you're looking for scale in the cloud

# In[17]:


# initialize pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY', 'YourAPIKeyIfNotSet'),  # find at app.pinecone.io
    environment=os.getenv('PINECONE_ENV', 'YourAPIKeyIfNotSet'),  # next to api key in console
)

index_name = "topic-modeling"

docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# # If you want to delete your vectors in your index to start over, run the code below!
# index = pinecone.Index(index_name)
# index.delete(delete_all='true')


# ### Option #2: Chroma
# 
# Use this if you're looking for local and easy to set up

# In[22]:


# load it into Chroma
docsearch = Chroma.from_documents(docs, embeddings)


# Then we are going to create a custom prompt for our Retriever. I'm doing this because the out of the [out-of-the-box](https://github.com/hwchase17/langchain/blob/7414e9d19603c962063dd337cdcf3c3168d4b8be/langchain/chains/question_answering/stuff_prompt.py#L20) prompt used here isn't bad, but a bit generic for my use case. Plus, I only really want to *answer a question* I want to generated a mini-summary based off of docs.
# 
# Let's switch it up!

# In[23]:


# The system instructions. Notice the 'context' placeholder down below. This is where our relevant docs will go.
# The 'question' in the human message below won't be a question per se, but rather a topic we want to get relevant information on
system_template = """
You will be given text from a podcast transcript which contains many topics.
You goal is to write a summary (5 sentences or less) about a topic the user chooses
Do not respond with information that isn't relevant to the topic that the user gives you
----------------
{context}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

# This will pull the two messages together and get them ready to be sent to the LLM through the retriever
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


# In[24]:


# I'm using gpt4 for the increased reasoning power.
# I'm also setting k=4 so the number of relevant docs we get back is 4. This parameter should be tuned to your use case
qa = RetrievalQA.from_chain_type(llm=llm4,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(k=4),
                                 chain_type_kwargs = {
#                                      'verbose': True,
                                     'prompt': CHAT_PROMPT
                                 })


# Then let's iterate through the topics that we found and run our QA query on them.
# 
# This will print out our expanded topics. This is the final result you can use wherever you want!

# In[25]:


# Only doing the first 3 for conciseness 
for topic in topics_structured[:5]:
    query = f"""
        {topic['topic_name']}: {topic['description']}
    """

    expanded_topic = qa.run(query)

    print(f"{topic['topic_name']}: {topic['description']}")
    print(expanded_topic)
    print ("\n\n")


# ## Bonus: Chapters With Timestamps
# 
# Because why not?
# 
# We have the timestamps on the transcript so let's pull them out and get timestamp chapters. This is helpful so you can scrub to the topic when you're listening.
# 
# I tried a few methods to do this, including function calling, but I found just a regular prompt worked great. It's not *that* hard of a task to pull out a timestamp. I did the Retrieval chain again to get relevant documents, then asked the LLM to pull out the earliest timestamp it saw a topic was talked about.
# 
# **Hardcore**: Right now this will pull out the timestamp of the monologue. However the topic may or may not start at the beginning, maybe it's the middle? Timestamps could be off. If you wanted to go more hardcore accurate you could go down to the word level and make a guestimate as to when the topic actually started.
# 
# Same as above, we'll make custom prompts for our QA chain

# In[26]:


system_template = """
What is the first timestamp when the speakers started talking about a topic the user gives?
Only respond with the timestamp, nothing else. Example: 0:18:24
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


# In[27]:


qa = RetrievalQA.from_chain_type(llm=llm4,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(k=4),
                                 chain_type_kwargs = {
#                                      'verbose': True,
                                     'prompt': CHAT_PROMPT
                                 })


# In[28]:


# Holder for our topic timestamps
topic_timestamps = []

for topic in topics_structured:

    query = f"{topic['topic_name']} - {topic['description']}"
    timestamp = qa.run(query)
    
    topic_timestamps.append(f"{timestamp} - {topic['topic_name']}")


# They might be out of order so let's sort them and print

# In[29]:


print ("\n".join(sorted(topic_timestamps)))


# Check out the audio for this episode [here](https://steno.ai/my-first-million/steph-smith-jobs-of-the-future-fractional-real-estate-mouth)
# 
# Awesome! This is great, what domain are you going to parse topics from? Please let me know on [Twitter](https://twitter.com/GregKamradt) or contact me directly at contact@dataindependent.com

# In[ ]:




