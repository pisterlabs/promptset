#%%
# read csv file 
import pandas as pd
import numpy as np
import os
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
#import openai
from dotenv import load_dotenv


load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
#%%
path = '/Volumes/boot420/Users/data/climate_network/cop22/'
file_user = 'users_cop22.csv'
file_tweets = 'tweets_cop22.pkl'

# read pickle file
cop = pd.read_pickle(os.path.join(path,file_tweets))

#%%
# only english 
cop = cop[cop['lang'] == 'en']

# remove retweets ( if starts euth RT)
cop = cop[~cop['text'].str.contains('RT')]
cop = cop[cop['referenced_type'].isna()]


def get_topics(df_cop):
    docs = df_cop['text'].tolist()

    docs = [re.sub(r"http\S+", "", doc) for doc in docs]



    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    model = BERTopic( 
                        vectorizer_model =   vectorizer_model,
                        ctfidf_model      =   ctfidf_model,
                        nr_topics        =  'auto',
                        min_topic_size   =   100,
                    )

    topics ,probs = model.fit_transform(docs)
    df_cop['topics'] = topics

    #cop.to_pickle(os.path.join(path,'tweets_cop22_topics.pkl'))
    #model.get_topic_info().to_csv(os.path.join(path,'topics_cop22.csv'))
    


    return df_cop





#%%
#cop['topics'] = topics











#%%

def get_topics_label(model):

    topics = list(model.get_topic_info().head(10)['Topic']) # get inferred topics 
    topic_words = model.get_topics() # get words for each topic
    labels = {}

    for topic in topics:
        tweets = model.get_representative_docs(topic)
        prompt = "you are a tweet labeler, you are given representative words from a topic and three representative tweets, give more weight to the words, given all these information give a short label for the topic (max 10 words), starts all with topic:."
        words = [word[0] for word in topic_words[topic]]


        topic_label = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": 'words'+ str(words)},
                    {"role": "user", "content": 'tweet'+ tweets[0]},
                    {"role": "user", "content": 'tweet'+ tweets[1]},
                    {"role": "user", "content": 'tweet'+ tweets[2]},
                ]
        )
        labels[topic] = topic_label['choices'][0]['message']['content']
    
    return labels
# %%

topics = list(model.get_topic_info().head(10)['Topic']) # get inferred topics 
topic_words = model.get_topics() # get words for each topic
labels = {}

for topic in topics:
    tweets = model.get_representative_docs(topic)
    prompt = "you are a tweet labeler, you are given representative words from a topic and three representative tweets, give more weight to the words, given all these information give a short label for the topic (max 10 words), starts all with topic:."
    words = [word[0] for word in topic_words[topic]]
    print('topic', topic)

    topic_label = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": 'words'+ str(words)},
                {"role": "user", "content": 'tweet'+ tweets[0]},
                {"role": "user", "content": 'tweet'+ tweets[1]},
                {"role": "user", "content": 'tweet'+ tweets[2]},
            ]
    )
    labels[topic] = topic_label['choices'][0]['message']['content']
# %%
# get 5 longest docs
longest_docs = sorted(docs, key=len, reverse=True)[:10]
# %%

