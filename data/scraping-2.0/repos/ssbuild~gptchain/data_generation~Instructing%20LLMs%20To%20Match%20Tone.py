#!/usr/bin/env python
# coding: utf-8

# # Instructing LLMs To Match Tone
# 
# LLMs that generate text are awesome, but what if you want to edit the tone/style it responds with?
# 
# We've all seen the [pirate](https://python.langchain.com/en/latest/modules/agents/agents/custom_llm_agent.html#:~:text=template%20%3D%20%22%22%22Answer%20the%20following%20questions%20as%20best%20you%20can%2C%20but%20speaking%20as%20a%20pirate%20might%20speak.%20You%20have%20access%20to%20the%20following%20tools%3A) examples, but it would be awesome if we could tune the prompt to match the tone of specific people?
# 
# Below is a series of techniques aimed to generate text in the tone and style you want. No single technique will likely be *exactly* what you need, but I guarantee you can iterate with these tips to get a solid outcome for your project.
# 
# But Greg, what about fine tuning? Fine tuning would likely give you a fabulous result, but the barriers to entry are too high for the average developer (as of May '23). I would rather get the 87% solution today rather than not ship something. If you're doing this in production and your differentiator is your ability to adapt to different styles you'll likely want to explore fine tuning.
# 
# If you want to see a demo video of this, check out the Twitter post. For a full explination head over to YouTube.
# 
# ### 4 Levels Of Tone Matching Techniques:
# 1. **Simple:** As a human, try and describe the tone you would like
# 2. **Intermediate:** Include your description + examples
# 3. **AI-Assisted:** Ask the LLM to extract tone, use their output in your next prompt
# 4. **Technique Fusion:** Combine multiple techniques to mimic tone
# 
# **Today's Goal**: Generate tweets mimicking the style of online personalities. You could customize this code to generate emails, chat messages, writing, etc.
# 
# First let's import our packages

# In[1]:


# LangChain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate

# Environment Variables
import os
from dotenv import load_dotenv

# Twitter
import tweepy

load_dotenv()


# Set your OpenAI key. You can either put it as an environment variable or in the string below

# In[2]:


openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet')


# We'll be using `gpt-4` today, but you can swap out for `gpt-3.5-turbo` if you'd like

# In[3]:


llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name='gpt-4')


# ## Method #1: Simple - Describe the tone you would like
# 
# Our first method is going to be simply describing the tone we would like.
# 
# Let's try a few exmaples

# In[4]:


prompt = """
Please create me a tweet about going to the park and eating a sandwich.
"""

output = llm.predict(prompt)
print (output)


# Not bad, but I don't love the emojis and I want it to use more conversational modern language.
# 
# Let's try again

# In[5]:


prompt = """
Please create me a tweet about going to the park and eating a sandwich.

% TONE
 - Don't use any emojis or hashtags.
 - Use simple language a 5 year old would understand
"""

output = llm.predict(prompt)
print (output)


# Ok cool! The tone has changed. Not bad but now I want it to sound like a specific person. Let's try Bill Gates:

# In[6]:


prompt = """
Please create me a tweet about going to the park and eating a sandwich.

% TONE
 - Don't use any emojis or hashtags.
 - Respond in the tone of Bill Gates
"""

output = llm.predict(prompt)
print (output)


# It's ok, I'd give the response a `C+` right now.
# 
# Let's give some example tweets so the model can better match tone/style.
# 
# `⭐ Important Tip: When you're giving examples, make sure to have the examples the same as the desired output format. Ex: Tweets > Tweets, Email > Email`. Don't do `Tweets > Email`

# ## Method #2: Intermediate - Specify your tone description + examples
# 
# Examples speak a thousand words. Let's pass a few along with our instructions to see how it goes
# 
# ### Get a users Tweets
# 
# Next let's grab a users tweets. We'll do this in a function so it's easy to pull them later

# Since we are live Tweets, you'll need to grather some Twitter api keys. You can get these on the [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard). The free tier is fine, but watch out for rate limits.

# In[7]:


# Replace these values with your own Twitter API credentials
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', 'YourAPIKeyIfNotSet')
TWITTER_API_KEY_SECRET = os.getenv('TWITTER_API_KEY_SECRET', 'YourAPIKeyIfNotSet')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', 'YourAPIKeyIfNotSet')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', 'YourAPIKeyIfNotSet')


# In[9]:


# We'll query 70 tweets because we end up filtering out a bunch, but we'll only return the top 12.
# We will also only use a subset of the top tweets later
def get_original_tweets(screen_name, tweets_to_pull=70, tweets_to_return=12):
    
    # Tweepy set up
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_KEY_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    tweets = []
    
    tweepy_results = tweepy.Cursor(api.user_timeline,
                                   screen_name=screen_name,
                                   tweet_mode='extended',
                                   exclude_replies=True).items(tweets_to_pull)
    
    # Run through tweets and remove retweets and quote tweets so we can only look at a user's raw emotions
    for status in tweepy_results:
        if not hasattr(status, 'retweeted_status') and not hasattr(status, 'quoted_status'):
            tweets.append({'full_text': status.full_text, 'likes': status.favorite_count})

    
    # Sort the tweets by number of likes. This will help us short_list the top ones later
    sorted_tweets = sorted(tweets, key=lambda x: x['likes'], reverse=True)

    # Get the text and drop the like count from the dictionary
    full_text = [x['full_text'] for x in sorted_tweets][:tweets_to_return]
    
    # Conver the list of tweets into a string of tweets we can use in the prompt later
    example_tweets = "\n\n".join(full_text)
            
    return example_tweets


# Let's grab Bill Gates tweets and use those as examples

# In[10]:


user_screen_name = 'billgates'  # Replace this with the desired user's screen name
users_tweets = get_original_tweets(user_screen_name)


# Let's look at a sample of Bill's tweets

# In[11]:


print(users_tweets)


# ### Pass the tweets as examples

# In[12]:


template = """
Please create me a tweet about going to the park and eating a sandwich.

% TONE
 - Don't use any emojis or hashtags.
 - Respond in the tone of Bill Gates

% START OF EXAMPLE TWEETS TO MIMIC
{example_tweets}
% END OF EXAMPLE TWEETS TO MIMIC

YOUR TWEET:
"""

prompt = PromptTemplate(
    input_variables=["example_tweets"],
    template=template,
)

final_prompt = prompt.format(example_tweets=users_tweets)

print (final_prompt)


# In[13]:


output = llm.predict(final_prompt)
print (output)


# Wow! Ok now that is starting to get somewhere. Not bad at all! Sounds like Bill is in the room with us now.
# 
# Let's see if we can refine it even more.

# ## Method #3: AI-Assisted: Ask the LLM help with tone descriptions
# 
# Turns out I'm not great at describing tone. Examples are a good way to help, but can we do more? Let's find out.
# 
# I want to have the model tell me what tone *it* sees, then use that output as an *input* to the final prompt where I ask it to generate a tweet.
# 
# Almost like reverse engineering tone.
# 
# Why don't I do this all in one step? You likely could, but it would be nice to save this "tone" description for future use. Plus, I don't want the model to take too many logic jumps in a single response.
# 
# I first thought, 'well... what are the qualities of tone I should have it describe?'
# 
# Then I thought, Greg, c'mon man, you know better than that, see if the LLM has a good sense of what tone qualities there are. Duh.
# 
# Let's see what are the qualities of tone we should extract

# In[14]:


prompt = """
Can you please generate a list of tone attributes and a description to describe a piece of writing by?

Things like pace, mood, etc.

Respond with nothing else besides the list
"""

how_to_describe_tone = llm.predict(prompt)
print (how_to_describe_tone)


# Ok great! Now that we have a solid list of ideas on how to instruct our language model for tone. Let's do some tone extraction!
# 
# I found that when I asked the model for a description of the tone it would be passive and noncommittal so I included a line in the prompt about taking an active voice 

# In[15]:


def get_authors_tone_description(how_to_describe_tone, users_tweets):
    template = """
        You are an AI Bot that is very good at generating writing in a similar tone as examples.
        Be opinionated and have an active voice.
        Take a strong stance with your response.

        % HOW TO DESCRIBE TONE
        {how_to_describe_tone}

        % START OF EXAMPLES
        {tweet_examples}
        % END OF EXAMPLES

        List out the tone qualities of the examples above
        """

    prompt = PromptTemplate(
        input_variables=["how_to_describe_tone", "tweet_examples"],
        template=template,
    )

    final_prompt = prompt.format(how_to_describe_tone=how_to_describe_tone, tweet_examples=users_tweets)

    authors_tone_description = llm.predict(final_prompt)

    return authors_tone_description


# Let's combine the tone description and examples to see what tone attributes the model assigned to Bill Gates

# In[19]:


authors_tone_description = get_authors_tone_description(how_to_describe_tone, users_tweets)
print (authors_tone_description)


# Great, now that we have Bill Gate's tone style, let's put those tone instructions in with the prompt we had before to see if it helps

# In[20]:


template = """
% INSTRUCTIONS
 - You are an AI Bot that is very good at mimicking an author writing style.
 - Your goal is to write content with the tone that is described below.
 - Do not go outside the tone instructions below
 - Do not use hashtags or emojis
 - Respond in the tone of Bill Gates

% Description of the authors tone:
{authors_tone_description}

% Authors writing samples
{tweet_examples}

% YOUR TASK
Please create a tweet about going to the park and eating a sandwich.
"""

prompt = PromptTemplate(
    input_variables=["authors_tone_description", "tweet_examples"],
    template=template,
)

final_prompt = prompt.format(authors_tone_description=authors_tone_description, tweet_examples=users_tweets)


# In[21]:


llm.predict(final_prompt)


# Hmm, better! Not wonderful.
# 
# Let's try out the final approach

# ## Method 4 - **Technique Fusion:** Combine multiple techniques to mimic tone
# 
# After a lot of experimentation I've found the below tips to be helpful
# 
# * **Don't reference the word 'tweet' in your prompt** - The model has an extremely strong bias towards what a 'tweet' is an will overload you with hashtags and emojis. Rather call it "a short public statement around 300 characters"
# * **Ask the LLM for similar sounding authors** - Whereas model bias on the word 'tweet' (point #1) isn't great, we can use it in our favor. Ask the LLM which authors the writing style sounds like, then ask the LLM to respond like that author. It's not great that the model is basing the tone off *another* person but it's a great 89% solution. I learned of this technique from [Scott Mitchell](https://twitter.com/mitchell360/status/1657909800389464064).
# * **Examples should be in the output format you want** - Everyone has a different voice. Twitter voice, email voice etc. Make sure that the examples you feed to the prompt are the same voice as the output you want. Ex: Don't exect a book to be written from twitter examples.
# * **Use the Language Model to extract tone** - If you are at a loss for words on how to describe the tone you want, have the language model describe it for you. I found I needed to tell the model to be opinionated, it was too grey-area before.
# * **Topics matter** - Have the model propose topics *first*, *then* give you a tweet. Not only is it better to have things the author would actually talk about, but it's also really good to keep the model on track by having it outline the topics *first* then respond
# 
# Let's first identify authors the model thinks the example tweets sound like, then we'll reference those later. Keep in mind this isn't a true classification exercise and the point isn't to be 100% correct on similar people, it's to get a reference to who the model *thinks* is similar so we can use that inuition for instructions later.

# In[22]:


def get_similar_public_figures(tweet_examples):
    template = """
    You are an AI Bot that is very good at identifying authors, public figures, or writers whos style matches a piece of text
    Your goal is to identify which authors, public figures, or writers sound most similar to the text below

    % START EXAMPLES
    {tweet_examples}
    % END EXAMPLES

    Which authors (list up to 4 if necessary) most closely resemble the examples above? Only respond with the names separated by commas
    """

    prompt = PromptTemplate(
        input_variables=["tweet_examples"],
        template=template,
    )

    # Using the short list of examples so save on tokens and (hopefully) the top tweets
    final_prompt = prompt.format(tweet_examples=tweet_examples)

    authors = llm.predict(final_prompt)
    return authors


# In[23]:


authors = get_similar_public_figures(users_tweets)
print (authors)


# Ok that's not that exciting! Becuase we used Bill Gates' example tweets. Trust me that it's better with less-known people. We'll try this more later.
# 
# At last, the final output. Let's bring it all together in a single prompt. Notice the 2 step process in the "your task" section below

# In[24]:


template = """
% INSTRUCTIONS
 - You are an AI Bot that is very good at mimicking an author writing style.
 - Your goal is to write content with the tone that is described below.
 - Do not go outside the tone instructions below

% Mimic These Authors:
{authors}

% Description of the authors tone:
{tone}

% Authors writing samples
{example_text}
% End of authors writing samples

% YOUR TASK
1st - Write out topics that this author may talk about
2nd - Write a concise passage (under 300 characters) as if you were the author described above
"""

method_4_prompt_template = PromptTemplate(
    input_variables=["authors", "tone", "example_text"],
    template=template,
)

# Using the short list of examples so save on tokens and (hopefully) the top tweets
final_prompt = method_4_prompt_template.format(authors=authors,
                                               tone=authors_tone_description,
                                               example_text=users_tweets)


# In[26]:


# print(final_prompt) # Print this out if you want to see the full final prompt. It's long so I'll omit it for now


# In[27]:


output = llm.predict(final_prompt)
print (output)


# After a ton of iteration, I'm actually happy with that. But let's see this thing spread it's wings on multiple people.
# 
# ## Extra Credit: Loop this process through many twitter accounts
# 
# Let's see what different twitter accounts sound like. Note, this will burn tokens so use at your own risk!

# In[30]:


results = {} # To store the results

# # Or if you just wanna see the results of the loop below you can open up this json
# import json
# with open("../data/matching_tone_samples.json", "r") as f:
#     tone_samples = json.load(f)
# print (tone_samples)


# In[ ]:


accounts_to_mimic = ['jaltma', 'lindayacc', 'ShaanVP', 'dharmesh', 'sweatystartup', 'levelsio', 'Suhail', \
                     'hwchase17', 'elonmusk', 'packyM', 'benedictevans', 'paulg', 'AlexHormozi', 'DavidDeutschOxf', \
                     'stephsmithio', 'sophiaamoruso']
                     

for user_screen_name in accounts_to_mimic:
    
    # Checking to see if we already have done the user. If so, move to the next one
    if user_screen_name in results:
        continue
    
    results[user_screen_name] = ""
    
    user_screenname_string = f"User: {user_screen_name}"
    print (user_screenname_string)
    results[user_screen_name] += user_screenname_string
    
    # Get their top tweets
    users_tweets = get_original_tweets(user_screen_name)
    
    # Get their similar authors
    authors = get_similar_public_figures(users_tweets)
    authors_string = f"Similar Authors: {authors}"
    print (authors_string)
    results[user_screen_name] += "\n" + authors_string
    
    # Get their tone description
    authors_tone_description = get_authors_tone_description(how_to_describe_tone, users_tweets)
    
    # Only printing the first four attributes to save space
    sample_description = authors_tone_description.split('\n')[:4]
    sample_decscription_string = f"Tone Description: {sample_description}"
    print(sample_decscription_string)
    results[user_screen_name] += "\n" + sample_decscription_string + "\n"
    
    
    # Bring it all together in a single prompt
    prompt = method_4_prompt_template.format(authors=authors,
                                             tone=authors_tone_description,
                                             example_text=users_tweets)
    
    output = llm.predict(prompt)
    results[user_screen_name] += "\n" + output
    
    print ("\n")
    print (output)
    print ("\n\n")

