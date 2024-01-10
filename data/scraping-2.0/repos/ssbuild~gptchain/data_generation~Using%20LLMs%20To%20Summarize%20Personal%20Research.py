#!/usr/bin/env python
# coding: utf-8

# # Using LLMs To Summarize Personal Research
# 
# Our goal is to have LLM aid us in generating interview quetions for someone. I find that I'm constantly trying to ramp up to a person's background and story when preparing to meet them.
# 
# There is a ton of awesome resources about a person online we can use
# 
# * Twitter Profiles
# * Websites
# * Other Interviews (YouTube or Text)
# 
# Let's bring all these together by first pulling the information and then generating questions or bullet points we can use as preparation.
# 
# First let's import our packages! We'll be using LangChain to help us interact with OpenAI

# In[1]:


# LLMs
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# Twitter
import tweepy

# Scraping
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# YouTube
from langchain.document_loaders import YoutubeLoader
# !pip install youtube-transcript-api

# Environment Variables
import os
from dotenv import load_dotenv

load_dotenv()


# You'll need a few API keys to complete the script below. It's modular so if you don't want to pull from Twitter feel free to leave those blank

# In[2]:


TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', 'YourAPIKeyIfNotSet')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', 'YourAPIKeyIfNotSet')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', 'YourAPIKeyIfNotSet')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', 'YourAPIKeyIfNotSet')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet')


# For this tutorial, let's pretend we are going to be interviewing [Elad Gil](https://eladgil.com/) since he has a bunch of content online
# 
# ### Pulling Data From Twitter
# Great, now let's set up a function that will pull tweets for us. This will help us get current events that the user is talking about. I'm excluding replies since they usually don't have a ton of high signal text from the user. This is the same code that was used in the [Twitter AI Bot tutorial](https://youtu.be/yLWLDjT01q8).

# In[3]:


def get_original_tweets(screen_name, tweets_to_pull=80, tweets_to_return=80):
    
    # Tweepy set up
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    # Holder for the tweets you'll find
    tweets = []
    
    # Go and pull the tweets
    tweepy_results = tweepy.Cursor(api.user_timeline,
                                   screen_name=screen_name,
                                   tweet_mode='extended',
                                   exclude_replies=True).items(tweets_to_pull)
    
    # Run through tweets and remove retweets and quote tweets so we can only look at a user's raw emotions
    for status in tweepy_results:
        if hasattr(status, 'retweeted_status') or hasattr(status, 'quoted_status'):
            # Skip if it's a retweet or quote tweet
            continue
        else:
            tweets.append({'full_text': status.full_text, 'likes': status.favorite_count})

    
    # Sort the tweets by number of likes. This will help us short_list the top ones later
    sorted_tweets = sorted(tweets, key=lambda x: x['likes'], reverse=True)

    # Get the text and drop the like count from the dictionary
    full_text = [x['full_text'] for x in sorted_tweets][:tweets_to_return]
    
    # Convert the list of tweets into a string of tweets we can use in the prompt later
    users_tweets = "\n\n".join(full_text)
            
    return users_tweets


# Ok cool, let's try it out!

# In[4]:


user_tweets = get_original_tweets("eladgil")
print (user_tweets[:300])


# Awesome, now we have a few tweets let's move onto pulling data from a web page or two.
# 
# ### Pulling Data From Websites
# 
# Let's do two pages
# 
# 1. His personal website which has his background - https://eladgil.com/
# 2. One of my favorite blog posts from him around AI defensibility & moats - https://blog.eladgil.com/p/defensibility-and-competition
# 
# First let's create a function that will scrape a website for us.
# 
# We'll do this by pulling the raw html, put it in a BeautifulSoup object, then convert that object to Markdown for better parsing

# In[5]:


def pull_from_website(url):
    
    # Doing a try in case it doesn't work
    try:
        response = requests.get(url)
    except:
        # In case it doesn't work
        print ("Whoops, error")
        return
    
    # Put your response in a beautiful soup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get your text
    text = soup.get_text()

    # Convert your html to markdown. This reduces tokens and noise
    text = md(text)
     
    return text


# In[6]:


# I'm going to store my website data in a simple string.
# There is likely optimization to make this better but it's a solid 80% solution

website_data = ""
urls = ["https://eladgil.com/", "https://blog.eladgil.com/p/defensibility-and-competition"]

for url in urls:
    text = pull_from_website(url)
    
    website_data += text


# Awesome, now that we have both of those data sources, let's check out a sample

# In[7]:


print (website_data[:400])


# Awesome, to round us off, let's get the information from a youtube video. YouTube has tons of data like Podcasts and interviews. This will be valuable for us to have.
# 
# ### Pulling Data From YouTube
# 
# We'll use LangChains YouTube loaders for this. It only works if there is a transcript on the YT video already, if there isn't then we'll move on. You could get the transcript via Whisper if you really wanted to, but that's out of scope for today.
# 
# We'll make a function we can use to loop through videos

# In[11]:


# Pulling data from YouTube in text form
def get_video_transcripts(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript


# In[13]:


# Using a regular string to store the youtube transcript data
# Video selection will be important.
# Parsing interviews is a whole other can of worms so I'm opting for one where Elad is mostly talking about himself
video_urls = ['https://www.youtube.com/watch?v=nglHX4B33_o']
videos_text = ""

for video_url in video_urls:
    video_text = get_video_transcripts(video_url)
    
    videos_text += video_text


# Let's look at at sample from the video

# In[14]:


print(video_text[:300])


# Awesome now that we have all of our data, let's combine it together into a single information block

# In[15]:


user_information = user_tweets + website_data + video_text


# Our `user_information` variable is a big messy wall of text. Ideally we would clean this up more and try to increase the signal to noise ratio. However for this project we'll just focus on the core use case of gathering data.
# 
# Next we'll chunk our wall of text into pieces so we can do a map_reduce process on it. If you want learn more about techniques to split up your data check out my video on [OpenAI Token Workarounds](https://www.youtube.com/watch?v=f9_BWhCI4Zo)

# In[16]:


# First we make our text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)


# In[17]:


# Then we split our user information into different documents
docs = text_splitter.create_documents([user_information])


# In[18]:


# Let's see how many documents we created
len(docs)


# Because we have a special requset for the LLM on our data, I want to make custom prompts. This will allow me to tinker with what data the LLM pulls out. I'll use Langchain's `load_summarize_chain` with custom prompts to do this. We aren't making a summary, but rather just using `load_summarize_chain` for its easy mapreduce functionality.
# 
# First let's make our custom map prompt. This is where we'll instruction the LLM that it will pull out interview questoins and what makes a good question.

# In[19]:


map_prompt = """You are a helpful AI bot that aids a user in research.
Below is information about a person named {persons_name}.
Information will include tweets, interview transcripts, and blog posts about {persons_name}
Your goal is to generate interview questions that we can ask {persons_name}
Use specifics from the research when possible

% START OF INFORMATION ABOUT {persons_name}:
{text}
% END OF INFORMATION ABOUT {persons_name}:

Please respond with list of a few interview questions based on the topics above

YOUR RESPONSE:"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "persons_name"])


# Then we'll make our custom combine promopt. This is the set of instructions that we'll LLM on how to handle the list of questions that is returned in the first step above.

# In[20]:


combine_prompt = """
You are a helpful AI bot that aids a user in research.
You will be given a list of potential interview questions that we can ask {persons_name}.

Please consolidate the questions and return a list

% INTERVIEW QUESTIONS
{text}
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "persons_name"])


# Let's create our LLM and chain. I'm increasing the color a bit for more creative language. If you notice that your questions have hallucinations in them, turn temperature to 0

# In[21]:


llm = ChatOpenAI(temperature=.25, model_name='gpt-4')

chain = load_summarize_chain(llm,
                             chain_type="map_reduce",
                             map_prompt=map_prompt_template,
                             combine_prompt=combine_prompt_template,
#                              verbose=True
                            )


# Ok, finally! With all of our data gathered and prompts ready, let's run our chain

# In[22]:


output = chain({"input_documents": docs, # The seven docs that were created before
                "persons_name": "Elad Gil"
               })


# In[23]:


print (output['output_text'])


# Awesome! Now we have some questions we can iterate on before we chat with the person. You can swap out different sources for different people.
# 
# These questions won't be 100% 'copy & paste' ready, but they should serve as a really solid starting point for you to build on top of.
# 
# Next, let's port this code over to a [Streamlit](https://streamlit.io/) app so we can share a deployed version easily
