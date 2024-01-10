
# %% [markdown]
# # Identifying commercial markets to filter them out from followers
# 
# The goal of this file is to be able to identify which of the followers that I
#  selected are commercial followers or otherwise small businesses. This 
# corrupts my input user base with non-people, so this is an attempt to remove
# these followers.
# %% [markdown]
# ## Imports

# %%
import json
import glob
import pickle
import collections
import random
from tqdm import tqdm as tqdm
import time
import config


import os
dirpath = os.path.dirname(os.path.realpath('__file__'))

import tweepy
import config

# NLP imports
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['https', 'http'])
import re
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# To check later if a words is in english or not. Note that to include some
# additional words as stop words, I just removed them from this dictionary
with open('./words_dictionary.json') as filehandle:
    words_dictionary = json.load(filehandle)
english_words = words_dictionary.keys()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# geo imports
import shapefile
from shapely.geometry import Point # Point class
from shapely.geometry import shape # shape() is a function to convert geo objects through the interface

# %% [markdown]

## All functions

# %%
def is_urban(pt, shapes):
    """
    Takes in a point and then checks to see if it lies within an urban boundary
    returns: Boolean
    """
    for boundary in shapes:
        if Point(pt).within(shape(boundary)): 
            return True
    return False

def handle_applier(handle):
    """
    Some twitter handles have the '@' symbol which shouldn't be 
    passed to the twitter API so we will remove them
    """
    if len(handle) == 0:
        return handle
    if handle[0] == '@':
        return handle[1:]
    return handle

def id_generator():
    """
    Generates user_id integers to supply to twitter while generating random
    users. Note that the docs say that these numbers are 64 bit unsigned
    integers (https://developer.twitter.com/en/docs/basics/twitter-ids)
    input: none
    output: a random 64-bit unsigned integer
    """
    # Pick bit length at random
    bit_length = np.random.randint(low=10, high=50)
    
    # Pick the bits at random
    digits = np.random.randint(low=0, high=2, size=bit_length)
    binary_number = ''.join(list(map(str, digits)))
    return int(binary_number, 2)

def get_user(tweet):
    """
    input: tweet dictionary
    returns: return the username
    """
    return tweet['user']['screen_name']


def get_hashtag_list(tweet):
    """
    input: tweet dictionary
    returns: list of all hashtags in both the direct tweet and the
    retweet 
    """

    l = []
    for d in tweet['entities']['hashtags']:
        l += [d['text']]

    if 'retweeted_status' in tweet.keys():
        for d in tweet['retweeted_status']['entities']['hashtags']:
            l += [d['text']]
    return l


def tokenizer_cleaner_nostop_lemmatizer(text):
    """
    This function tokenizes the text of a tweet, cleans it off punctuation,
    removes stop words, and lemmatizes the words (i.e. finds word roots to remove noise)
    I am largely using the gensim and spacy packages 

    Input: Some text
    Output: List of tokenized, cleaned, lemmatized words
    """

    tokenized_depunkt = gensim.utils.simple_preprocess(text, min_len=4, deacc=True)
    tokenized_depunkt_nostop = ([word for word in tokenized_depunkt 
                                 if (word not in stop_words and word in english_words)])
    
    # Lemmatizer while also only allowing certain parts of speech.
    # See here: https://spacy.io/api/annotation
    allowed_pos = ['ADJ', 'ADV', 'NOUN', 'PROPN','VERB']
    doc = nlp(' '.join(tokenized_depunkt_nostop))
    words_final = [token.lemma_ for token in doc if token.pos_ in allowed_pos]
    return words_final

    
def get_tweet_words_list(tweet):
    """
    This function takes in a tweet and checks if there is a retweet associated 
    with it. It then returns a list of tokenized words without punctuation.
    input: tweet
    output: list of tokenized words without punctuation
    """

    text = tweet['full_text']
    clean_words = tokenizer_cleaner_nostop_lemmatizer(text)
    
    if 'retweeted_status' in tweet.keys():
        retweet_text = tweet['retweeted_status']['full_text']
        retweet_clean_words = tokenizer_cleaner_nostop_lemmatizer(retweet_text)
        clean_words += retweet_clean_words
    return clean_words

# %% [markdown]
# ## Loading the data and adding an `is_urban` column
# 
# We get the data of urban regions from US Census data, and we write a simple
# for loop to see if a particular city lies in an urban region. I only want to
# select users from urban areas.

# %%
load_columns = ['Twitter', 'city', 'State', 'x', 'y', 'zip' ]
markets = pd.read_csv('./farmers_market_twitter_all.csv', usecols=load_columns)
markets = markets.dropna(axis=0)
markets['coords'] = markets.apply(lambda row: (row['x'], row['y']), axis=1)


# %%
# shp = shapefile.Reader('./data/tl_2019_us_uac10/tl_2019_us_uac10.shp')
# shapes = shp.shapes() # get all the polygons


# %%
# tqdm.pandas()
# markets['is_urban'] = markets.progress_apply(lambda row: is_urban(row['coords'],
#                                                             shapes), axis=1)
# markets = markets[markets['is_urban']] # select only the urban_areas
# print('Number of eligible markets:', markets.shape[0])
# # export to csv so I don't have to do this again
# markets.to_csv('./data/markets_is_urban.csv', index=False)

markets = pd.read_csv('./data/markets_is_urban.csv')

# %% [markdown]
# ### I still need to further process the twitter handle because they are all 
# over the place 

# %%

# Remove all the '#NAME?' entries. Decreases the number of markets to 789.
markets = markets[markets['Twitter'] != '#NAME?']
# Only extract the bits after the last slash
markets['Twitter'] = markets['Twitter'].apply(lambda r: r.rsplit('/', 1)[-1])
markets['Twitter'] = markets.apply(lambda r: handle_applier(r['Twitter']), 
                                                            axis=1)

# %% [markdown]
# ## Download upto a 1000 tweets for farmers markets that lie in urban areas.

# %%
consumer_key = config.consumer_key
consumer_secret = config.consumer_secret
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# %% [markdown]

# ## Downloading 1000 tweets for each eligible market

# %%
# In future runs, if you don't have to download this data again,
# just load the original pickle file
# market_tweets_dict = {}
# for market in tqdm(markets['Twitter']):
#     try:
#         tweets = tweepy.Cursor(api.user_timeline,
#                                 screen_name=market,
#                                 tweet_mode='extended',
#                                 count=1000).items(1000)
#         for tweet in tweets:
#             market_tweets_dict[market] = market_tweets_dict.get(market, []) + [tweet._json]
#     except:
#         pass

# # Write file to disk
# with open('./data/market_tweets_dict.data', 'wb') as filehandle:
#     pickle.dump(market_tweets_dict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

# print('Number of markets:', len(list(market_tweets_dict.keys())))

# %% [markdown]

## Selection and labeling of farmers market tweets

# Here, we wish to first select only those markets that have tweeted over 500
# times. If it isn't, we will drop that market. If it is, we will transfer it 
# over into a dictionary of the form

# ```
# {
#     screen_name: {
#                     'tweets': [{tweet1_json}, ..., {tweet}],
#                     'label': 1
#                  }
# }
# ```


# %%
with open('./data/market_tweets_dict.data', 'rb') as filehandle:
    market_tweets_dict = pickle.load(filehandle)

dataset_tweets_random_users = {}
for market in market_tweets_dict:
    dataset_tweets_random_users[market] = {}
    
    if len(market_tweets_dict[market]) >= 500: # If the market has >=500 tweets
        dataset_tweets_random_users[market]['tweets'] = market_tweets_dict[market]
        dataset_tweets_random_users[market]['label'] = 1
# %% [markdown]
# We still have about 485 valid markets with over 500 tweets, so this is still 
# something to work with. Next we find random accounts from 
# different geographic locations across the country to add the datasets 
# labeled 0. 

# %% [markdown]
# ### Finding random users to label as 0.

# We generate a user_id at random and download up to 500 tweets. We then check
# if there are infact 500 tweets, and if the user has at least 300 followers. 
# If not, we pass on that user. If yes, we add the user to `dataset_tweets`
# using the appropriate dictionary structure. We do this until we have 500 
# random users.

# %%
# num_selected = 0
# while num_selected < 500:
#     random_id = id_generator()
#     try:
#         user = api.get_user(user_id=random_id,
#                                     lang='en',
#                                     include_entities=True)
#         # tweets = tweepy.Cursor(api.user_timeline,
#         #                         user_id=random_id,
#         #                         tweet_mode='extended',
#         #                         count=500).items(500)
#         user = user._json
#         if user['followers_count'] >= 300 and user['statuses_count'] >= 500:
#             num_selected += 1
#             print(num_selected)
#             screen_name = user['screen_name']
#             dataset_tweets_random_users[screen_name] = {}
#             dataset_tweets_random_users[screen_name]['label'] = 0
#             tweets = tweepy.Cursor(api.user_timeline,
#                                     user_id=random_id,
#                                     tweet_mode='extended',
#                                     count=500).items(500)
#             for tweet in tweets:
#                 dataset_tweets_random_users[screen_name]['tweets'] = (
#                     dataset_tweets_random_users[screen_name].get('tweets', []) 
#                                                                 + [tweet._json])
#     except:
#         pass

# with open('./data/dataset_tweets_random_users.data', 'wb') as filehandle:
#     pickle.dump(dataset_tweets_random_users, filehandle, 
#                 protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/dataset_tweets_random_users.data', 'rb') as filehandle:
    dataset_tweets_random_users = pickle.load(filehandle)

# %% [markdown]

#  ## Building more suitable dictionary format to ingest later.

# Note that the structure of the `dataset_tweets_random_users` dictionary is 
# ```
# {
#     screen_name: {
#                     'tweets': [{tweet1_json}, ..., {tweet}],
#                     'label': 1
#                  }
# }
# ```

# We now want to build all the usual data cleaning tokenziation, lemmatization
# pipeline and then build the corpus of words. For this, I am just copying and
# functions from `LDA.ipynb`. Ideally, I would want to have a `utils` file and
# then just import functions as needed from `utils.py`. With more time, I can
# probably clean up my code base significantly. Dividing the code into those 
# that download data and then export things a dictionaries, and those that 
# utilize functions from `utils.py` to keep everything clean and modular.

# I finally want a dictionary of the form

# ```
# {
#     user: {
#               'hashtags': [..., ..., ...],
#               'full_text': [cleaned, tokenized, lemmatized, words of tweets],
#               'label': 0 or 1 
#           }
# }
# ```

# We have defined some utility functions above to make this easier - this has
# been copied over from `LDA.py`. Ideally I want these in a utils file so that
# I can just import from there rather than repeating functions.

# %%
lda_dict_random_users = {}
counter = 0
for user in tqdm(dataset_tweets_random_users):
    try:
        lda_dict_random_users[user] = {}
        lda_dict_random_users[user]['hashtags'] = []
        lda_dict_random_users[user]['fulltext'] = []

        
        tweets = dataset_tweets_random_users[user]['tweets']
        label = dataset_tweets_random_users[user]['label']

        for tweet in tweets:
            hashtags = get_hashtag_list(tweet)
            words = get_tweet_words_list(tweet)

            lda_dict_random_users[user]['hashtags'].extend(hashtags)
            lda_dict_random_users[user]['fulltext'].extend(words)
            lda_dict_random_users[user]['label'] = label
    except:# Just skip the user if something goes wrong
        pass

# if you want to write the data to disk.

with open('./data/lda_dict_random_users.data', 'wb') as filehandle:
    pickle.dump(lda_dict_random_users, filehandle, 
                protocol=pickle.HIGHEST_PROTOCOL)
