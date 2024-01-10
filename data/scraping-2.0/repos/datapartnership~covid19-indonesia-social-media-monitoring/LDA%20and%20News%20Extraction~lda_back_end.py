"""
This script runs an LDA on Twitter, Instagram and news data.

"""
#Import packages
import numpy as np
import pandas as pd
from dateutil.parser import parse
import string
from nltk.corpus import stopwords
import nltk
import re
import os
import multiprocessing as mp
from multiprocessing import Pool
from collections import Counter
#nltk.download('stopwords')
import plotly.express as px
import gensim 
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import STOPWORDS
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib
from wordcloud import WordCloud, STOPWORDS
from tqdm import tqdm
#Get the other modules
from run_lda import run_lda

#Define Datasets
dataset = ["/mnt/alicia/geo_tweets.csv", "/mnt/alicia/hashtag_tweets.csv", #Twitter Data
"/mnt/alicia/indonesian_news/clean_indonesia_news.csv", #News Data
"/mnt/alicia/instagram_data.csv"] #Instagram Data

#Loop through the datasets, get word cloud and graph information.
def main():
    for i in tqdm(dataset):
         # Convert the dates
        df = pd.read_csv(i)
        print("dataset loaded")
        df["date"]= pd.to_datetime(df["date"], errors = 'coerce').dt.date
        print("dates parsed")
        #Drop columns with bad dates and text
        df = df.dropna(subset=['date'])
        df = df.dropna(subset=['text'])
        #Ensure start date
        startdate = pd.to_datetime("2020-03-24").date()
        df = df[df["date"] > startdate]
        print(df.shape)
        sample_size = df.shape[0]
        # Convert to list
        word_docs = df["text"].values.tolist()
        word_docs_words = word_docs
        word_docs_words = list(sent_to_words(word_docs))
        print("Sent to words worked")
        #Stopwords
        stops = stopwords.words('english') + stopwords.words('indonesian') + other_stops
        # Build the bigram and trigram models
        #bigram = gensim.models.Phrases(word_docs_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        #trigram = gensim.models.Phrases(bigram[word_docs_words], threshold=100)  
        #bigram_mod = gensim.models.phrases.Phraser(bigram)
        #trigram_mod = gensim.models.phrases.Phraser(trigram)
        #Remove Stops
        data_ready = [[word for word in simple_preprocess(str(doc)) if word not in stops] for doc in word_docs_words]
        # Create Dictionary
        print("Data is ready")
        id2word = corpora.Dictionary(data_ready)
        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_ready]
        print("Corpus created")
        #Run the LDA
        lda_model = run_lda(corpus, id2word)
        print("Lda run")
        topics = lda_model.show_topics(formatted=False)
        graph_grouped = lda_over_time(df, lda_model, corpus)
        if i == "/mnt/alicia/geo_tweets.csv":
            #Time series graph
            graph_grouped.to_csv("/mnt/alicia/geo_graph_grouped.csv", index=False)
            #1:1 mapping
            #df.to_csv("geo_tweets_dominant_topic_df.csv")
            with open("/mnt/alicia/geo_word_cloud_topics.txt", "w") as output:
                output.write(str(topics))
            with open("/mnt/alicia/geo_sample_size.txt", "w") as output:
                output.write(str(sample_size))
        elif i == "/mnt/alicia/hashtag_tweets.csv":
            #Time series graph
            graph_grouped.to_csv("/mnt/alicia/hashtag_graph_grouped.csv", index=False)
            #df.to_csv("hashtag_tweets_dominant_topic_df.csv")
            with open("/mnt/alicia/hashtag_word_cloud_topics.txt", "w") as output:
                output.write(str(topics))
            with open("/mnt/alicia/hashtag_sample_size.txt", "w") as output:
                output.write(str(sample_size))
        elif i == "/mnt/alicia/indonesian_news/clean_indonesia_news.csv":
            #Time series graph
            graph_grouped.to_csv("/mnt/alicia/news_graph_grouped.csv", index=False)
            #df.to_csv("news_dominant_topic_df.csv")
            with open("/mnt/alicia/news_word_cloud_topics.txt", "w") as output:
                output.write(str(topics))
            with open("/mnt/alicia/news_sample_size.txt", "w") as output:
                output.write(str(sample_size))
        elif i == "/mnt/alicia/instagram_data.csv":
            #Time series graph
            graph_grouped.to_csv("/mnt/alicia/instagram_graph_grouped.csv", index=False)
            #df.to_csv("news_dominant_topic_df.csv")
            with open("/mnt/alicia/instagram_word_cloud_topics.txt", "w") as output:
                output.write(str(topics))
            with open("/mnt/alicia/instagram_sample_size.txt", "w") as output:
                output.write(str(sample_size))

#Other Helpers
#Define some functions
def sent_to_words(sentences):
    for sent in tqdm(sentences):
        sent = re.sub(r'\S*@\S*\s?', '', str(sent))  # remove emails
        sent = re.sub(r'\s+', ' ', str(sent))  # remove newline chars
        sent = re.sub(r"\'", "", str(sent))  # remove single quotes
        sent = re.sub(r'http\S+', '', str(sent)) #remove links
        sent = re.sub(r'\b\w{1,3}\b', '', str(sent)) #remove short words
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

#Get the non-1:1 groupings
def lda_over_time(df, lda_model, corpus):
    #Produce Topics
    graph_topics = []
    to_append = []
    graph_topics = []
    x=lda_model.show_topics(num_topics=8, num_words=10,formatted=False)
    topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
    #Below Code Prints Only Words 
    for topic,words in topics_words:
        graph_topics.append(", ".join(words))
    for i in range(0, df.shape[0]):
        to_append.append(lda_model[corpus[i]])
    df["scores"] = to_append
    score = df['scores'].apply(pd.Series).applymap(lambda x: x[1] if isinstance(x, tuple) else 0)
    df = df.merge(score, left_index= True, right_index= True)
    df = df[["date", 0, 1, 2, 3, 4, 5, 6, 7]]
    new_cols = ["date"] + graph_topics
    df.columns = new_cols
    Topics = df.columns[1:8]
    # Convert the dates
    df["date"] = pd.to_datetime(df['date']).dt.date
    df['Week'] = pd.to_datetime(df.date) - pd.to_timedelta(7, unit='d')
    grouped = df.groupby(pd.Grouper(key='date'))[Topics].sum().reset_index().sort_values('date')
    graph_grouped = pd.melt(grouped, id_vars=['date'], value_vars=Topics)
    graph_grouped.columns = ["date", "Topic", "Sum"]
    return(graph_grouped)

#1:1 Mapping of Dominant Topics
def format_topics_sentences(ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(tqdm(ldamodel[corpus])):
            row = row_list[0] if ldamodel.per_word_topics else row_list            
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)
        
#Define some other stops
other_stops = ['gw', 'yg', 'rt','rt', 'also', 'may', 'us', 'said', 'wa', 'ha', '-', '_', "–", 'say', 'u', 'dy', 'rt', '1', '2', 'like', '01', '02', '03'
            'aja', 'ya', 'ga', 'semoga', 'kalo', 'tdk', 'amin', 'orang', 'udah', '...', 't…', 'please', 'em', '…', 'ya', 'yaa', 'amin', 'amp', 'orang', 
            'al', 'go', 'done', 'good', 'time', 'amp', 'new', 'btstwt', 'day', 'world', 'cc', 'klo', 'dungu', 'si', 'jae', 'ya', 'kang', 'emojis', 'rt' + 
            'hati', 'anak', 'kau', 'mata', 'sehat', 'turun', 'mati', 'hidup', 'akal', 'air', 'salah', 'uangnya', 'dunia', 'cinta', 'pakai', 'manusia', 'ya',
            'yaa', 'amin', 'amp', 'orang', 'al', 'tdk', 'amp', 'utk', 'dgn', 'sdh', 'krn', 'sy', 'dg', 'medium', 'dr', 'org', 'mrk', 'ya', 'aja', 'gak', 'ga',
            'kalo', 'udah', 'akun', 'nya', 'biar', 'nih', 'tau', 'sih', 'follow', 'tagar', 'banget', 'orang', 'gue', 'lu', 'pake', 'tertjemplung', 'dah', 'gitu',
            'tuh', 'emang', 'jgn', 'bilang', 'bro', 'si', 'semangat', 'selamat', 'akbar', 'terima', 'alhamdullilah', 'bang', 'pagi', 'amp', 'salam', 'aja', 'ya',
            'ga', 'semoga', 'kalo', 'tdk', 'amin', 'orang', 'udah', 'bn', 'etc', 'c', 'wa','hr','k', 'cc', 'bu', 'jae', 'lo', 'gua', 'si', 'jae', 'ya', 'clo', 'ya',
            'aja', 'gak', 'ga', 'kalo', 'udah', 'akun', 'nya', 'biar', 'nih', 'tau', 'sih', 'follow', 'tagar', 'banget', 'orang', 'gue', 'lu', 'pake', 'tertjemplung',
            'dah', 'gitu', 'tuh', 'emang', 'jgn', 'bilang', 'bro', 'si', 'semangat', 'selamat', 'akbar',  'terima', 'alhamdullilah', 'bang', 'pagi', 'amp', 'salam',
            'hati', 'anak', 'kau', 'mata', 'sehat', 'turun', 'mati', 'hidup', 'akal', 'air', 'salah', 'uangnya', 'dunia', 'cinta', 'pakai', 'manusia',
            'ya', 'yaa', 'amin', 'amp', 'orang', 'al', 'go', 'done', 'good', 'time', 'amp', 'new', 'btstwt', 'day', 'world' , 'cc', 'klo', 'dungu', 'si', 'jae', 'ya',
            'kang', 'emojis', 'gua', 'bong', 'non', 'wkwkwk', 'wkwk', 'haha', 'hahaha', 'hahahaha']

if __name__ == "__main__":
    main()