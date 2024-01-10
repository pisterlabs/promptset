import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

# import libraries
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import re,random,os
import seaborn as sns
from nltk.corpus import stopwords
import string
from pprint import pprint as pprint

# spacy for basic processing, optional, can use nltk as well(lemmatisation etc.)
import spacy

#gensim for LDA
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#plotting tools
import pyLDAvis
import pyLDAvis.gensim #dont skip this
import matplotlib.pyplot as plt


# tokenize using gensims simple_preprocess
def sent_to_words(sentences, deacc = True):  # deacc = True removes punctuations
    for sentence in sentences:
        '''
        Return sends a specified value back to its caller whereas Yield can produce a sequence of values. 
        We should use yield when we want to iterate over a sequence, but don’t want to store the entire sequence in memory. 
        Yield is used in Python generators. 
        '''
        yield(simple_preprocess(str(sentence)))


# compute coherence value at various values of alpha and num_topics
def compute_coherence_values(dictionary, corpus, texts, num_topics_range,alpha_range):
    coherence_values=[]
    model_list=[]
    for alpha in alpha_range:
        for num_topics in num_topics_range:
            lda_model= gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, alpha=alpha,num_topics=num_topics,\
                                                      per_word_topics=True)
            model_list.append(lda_model)
            coherencemodel=CoherenceModel(model=lda_model,texts=texts,dictionary=dictionary,coherence='c_v')
            coherence_values.append((alpha,num_topics,coherencemodel.get_coherence()))
    return model_list,coherence_values

# functions for removing stopwords and lemmatization
def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts,allowed_postags=['NOUN','ADJ','VERB','ADV']):
    """https://spacy.io/api/annotation"""
    texts_out=[]
    # Initialize spacy english model 
    # Do not need dependency parsing nor entity recognition
    nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])

    for sent in texts:
        doc = nlp(' '.join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# plot
def plot_coherence(coherence_df,alpha_range,num_topics_range):
    plt.figure(figsize=(16,6))
    
    for i,val in enumerate(alpha_range):
        #subolot 1/3/i
        plt.subplot(1,3,i+1)
        alpha_subset=coherence_df[coherence_df['alpha']==val]
        plt.plot(alpha_subset['num_topics'],alpha_subset['coherence_value'])
        plt.xlabel('num_topics')
        plt.ylabel('Coherence Value')
        plt.title('alpha={0}'.format(val))
        plt.ylim([0.30,1])
        plt.legend('coherence value', loc='upper left')
        plt.xticks(num_topics_range)

def gen_dict_and_corpus(input_df, data_column = "Text", sample_index = 1):
    # Convert text corpus to list
    data = input_df[data_column].values.tolist()
    data_words = list(sent_to_words(data))

    # First 10 words of random sample
    print(data_words[sample_index][0:10])

    stop_words = stopwords.words('english') + list(string.punctuation)

    # Remove stop words
    data_words_npstops = remove_stopwords(data_words, stop_words)

    # lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_npstops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(data_lemmatized[sample_index][0:10])

    # Compare the nostop, lemmatised version with the original one
    print(' '.join(data_words[sample_index][0:20]), '\n')
    print(' '.join(data_lemmatized[sample_index][0:20]))

    # Create dictionary and corpus

    # Create dictionary (id for each word)
    id2word = corpora.Dictionary(data_lemmatized)
    # Create corpus (term-frequency)
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    print(corpus[sample_index])

    # human-readable format of corpus (term-frequency)
    print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[0:2]])

    return id2word, corpus


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame(columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = row[0]
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # print(row)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                # print(topic_keywords)
                # print(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]))
                details = [int(topic_num), round(prop_topic,4), topic_keywords]
                sent_topics_df.loc[len(sent_topics_df)] = details
            else:
                break

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
