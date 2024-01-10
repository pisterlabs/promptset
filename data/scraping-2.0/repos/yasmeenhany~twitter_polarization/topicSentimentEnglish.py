from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re
import numpy as np
import pandas as pd
import csv
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, phrases

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
# NLTK Stop words
from nltk.corpus import stopwords
import os

def main():
    # uncomment the file that we want to assign topic + sentiment to
    df_csv = pd.read_csv("dataset/phase1/non_replies.csv", encoding='ISO-8859-1')
    # df_csv = pd.read_csv("dataset/phase1/replies.csv", encoding='ISO-8859-1')
    # df_csv = pd.read_csv("dataset/phase1/replied_to.csv", encoding='ISO-8859-1')
    # df_csv = pd.read_csv("dataset/phase2/non_replies.csv", encoding='ISO-8859-1')
    # df_csv = pd.read_csv("dataset/phase2/replies.csv", encoding='ISO-8859-1')
    # df_csv = pd.read_csv("dataset/phase2/replied_to.csv", encoding='ISO-8859-1')
    # df_csv = pd.read_csv("dataset/phase3/non_replies.csv", encoding='ISO-8859-1')
    # df_csv = pd.read_csv("dataset/phase3/replies.csv", encoding='ISO-8859-1')
    # df_csv = pd.read_csv("dataset/phase3/replied_to.csv", encoding='ISO-8859-1')
    df_csv.head()
    textList = df_csv.values.tolist()
    print(len(textList))
    text = ""
    #                                                   ------- Loading dataset files -------

    # Uncomment the block corresponding the phase that we want to assign topics and sentiment to only
    # Uncomment entire phase block if in training phase. If in assigning phase uncomment only the file we want to assign to

    # Phase 1

    # with open("dataset/phase1/non_replies.csv", encoding='ISO-8859-1') as csvfile:
    #     text= csvfile.read() # uncomment for either training or assigning phase
    # with open("dataset/phase1/replies.csv", encoding='ISO-8859-1') as csvfile:
    #     # text+= csvfile.read() # uncomment for training phase
    #      text = csvfile.read() # uncomment for assigning phase
    # with open("dataset/phase1/replied_to.csv", encoding='ISO-8859-1') as csvfile:
    #     # text+= csvfile.read() #uncomment for training phase
    #     text = csvfile.read() #uncomment for assigning phase

    # Phase 2

    # with open("dataset/phase2/non_replies.csv", encoding='ISO-8859-1') as csvfile:
    #     text= csvfile.read() # uncomment for either training or assigning phase
    # with open("dataset/phase2/replies.csv", encoding='ISO-8859-1') as csvfile:
    #     # text+= csvfile.read() # uncomment for training phase
    #      text = csvfile.read() # uncomment for assigning phase
    # with open("dataset/phase2/replied_to.csv", encoding='ISO-8859-1') as csvfile:
    #     # text+= csvfile.read() #uncomment for training phase
    #     text = csvfile.read() #uncomment for assigning phase

    # Phase 3

    # with open("dataset/phase3/non_replies.csv", encoding='ISO-8859-1') as csvfile:
    #     text= csvfile.read() # uncomment for either training or assigning phase
    # with open("dataset/phase3/replies.csv", encoding='ISO-8859-1') as csvfile:
    #     # text+= csvfile.read() # uncomment for training phase
    #      text = csvfile.read() # uncomment for assigning phase
    # with open("dataset/phase3/replied_to.csv", encoding='ISO-8859-1') as csvfile:
    #     # text+= csvfile.read() #uncomment for training phase
    #     text = csvfile.read() #uncomment for assigning phase

     #                                                   ------- Generating corpus -------
    nlp = spacy.load("en_core_web_sm")

    my_stop_words = ['https','co','from','text', 'subject', 're', 'edu', 'use','RT','make', 'jerusalemembassy', 'jerusalem', 'Jerusalem', 'amp', 'JerusalemEmbassy', 'usembassyjerusalem']
    for stopword in my_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True
    nlp.max_length= 1547045
    doc = nlp(text)
    texts, article = [], []
    for w in doc:
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and not w.like_url and w.is_ascii and not w.is_left_punct and not w.is_right_punct and w.lang_=='en' and w.is_alpha :
            article.append(w.lemma_)

        if w.text == '\n':
            texts.append(article)
            article= []

    texts = [x for x in texts if x != []]

    bigram = phrases.Phrases(texts)
    texts = [bigram[line] for line in texts]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    #                                                   ------- create LDA model (if in training phase) -------

    mallet_path = os.path.join('C:\\', 'new-mallet', 'mallet-2.0.8', 'bin', 'mallet.bat')
  
    # ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=dictionary)

    # ldamallet.save("phaseOne_full_model")
    # ldamallet.save("phaseTwo_full_model")
    # ldamallet.save("phaseThree_full_model")

    #                                                   ------- Loading LDA model (if in assigning phase) -------

    ldamallet = gensim.models.wrappers.LdaMallet.load("LDAmodels\\phaseOne_full_model")
    # ldamallet = gensim.models.wrappers.LdaMallet.load("LDAmodels\\phaseTwo_full_model")
    # ldamallet = gensim.models.wrappers.LdaMallet.load("LDAmodels\\phaseThree_full_model")


    #                                                   ------- Assigning topics to each text -------

    def format_topics_sentences(ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()
        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            dom_topic = ""
            perc_contrib = ""
            keywords = ""
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0 or j==1:  # => top 2 dominant topics
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    dom_topic+=str(topic_num)
                    perc_contrib+= str(round(prop_topic,4))
                    keywords+=topic_keywords
                else:
                    sent_topics_df = sent_topics_df.append(pd.Series([dom_topic, perc_contrib, keywords]), ignore_index=True)
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)
        
    #                                                   ------- Assigning sentiment to each text -------
    def getSentiment():
        analyzer = SentimentIntensityAnalyzer()
        sentimentResults= []
        for text in textList:
            for tweet in text:
                tweet = str(tweet)
                vs = analyzer.polarity_scores(tweet)
                if (vs['compound'] > 0.1):
                    sentimentResults.append("positive")
                elif (vs['compound'] < -0.1):
                    sentimentResults.append("negative")
                else:
                    sentimentResults.append("neutral")
        return sentimentResults
    #                                                   ------- Calling topic and sentiment assignment functions -------
    df_topic_sents_keywords = pd.DataFrame()
    df_topic_sents_keywords= format_topics_sentences(ldamodel=ldamallet, corpus=corpus, texts=textList)
    sentimentColumn= pd.Series(getSentiment())
    #                                                   ------- Adding sentiment column to full dataframe -------
    df_topic_sents_keywords= pd.concat([df_topic_sents_keywords,sentimentColumn], axis = 1)
    
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Sentiment']
    
     #                                                   ------- Saving dataframe to pickle  -------
     
     # Uncomment the line corresponding to what file was assigned topics and sentiment to

    # df_dominant_topic.to_pickle("topic_sentiment_pickles\\phase1_repliedto")
    # df_dominant_topic.to_pickle("topic_sentiment_pickles\\phase1_replies")
    # df_dominant_topic.to_pickle("topic_sentiment_pickles\\phase1_tweets")
    # df_dominant_topic.to_pickle("topic_sentiment_pickles\\phase2_repliedto")
    # df_dominant_topic.to_pickle("topic_sentiment_pickles\\phase2_replies")
    # df_dominant_topic.to_pickle("topic_sentiment_pickles\\phase2_tweets")
    # df_dominant_topic.to_pickle("topic_sentiment_pickles\\phase3_repliedto")
    # df_dominant_topic.to_pickle("topic_sentiment_pickles\\phase3_replies")
    # df_dominant_topic.to_pickle("topic_sentiment_pickles\\phase3_tweets")

if __name__ == "__main__":
    main()