import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import string
import re
from nltk.corpus import stopwords


# read file 
print("reading file")
generated_data_df = pd.read_csv("results_texts.csv", index_col = 0, header = 0)
print(generated_data_df.head(30))


# sentences to words
print("sentences to words")
for i in range(generated_data_df.shape[1]):
    # deacc=True removes punctuations => REMOVE PUNCTUATIONS
    generated_data_df.iloc[:,i] = [gensim.utils.simple_preprocess(str(sentence), deacc=True) for sentence in generated_data_df.iloc[:,i]]
print(generated_data_df.head())


# preprocess remove "the world is"
print("remove the world is")
for i in range(generated_data_df.shape[1]):
    generated_data_df.iloc[:,i] = [sentence[3:] for sentence in generated_data_df.iloc[:,i]]
print(generated_data_df.head())


# learn uni-grams and bigrams
print("learn uni-grams and bigrams")
data_words = generated_data_df.values.reshape(1,-1)[0].tolist()
print(len(data_words))
# Build the bigram and trigram models
print("Build the bigram and trigram models ")
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=1) # higher threshold fewer phrases.
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_model = gensim.models.phrases.Phraser(bigram)
print(list(bigram.vocab.keys())[:40])


# method extract n-grams and count frequency
def most_frequent_words(data_words, bigram_model):
    # remove stop words
    def remove_stopwords(texts):
        stop_words = [ 'i', 'me', 'my', 'we', 'you', "you're", "you've", "you'll", "you'd", 'he', 'him', 'she', "she's",  'it', "it's", 'they', 'them', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
        # stop_words = []
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    data_words_nostops = remove_stopwords(data_words)
    # extract 1-grams and 2-grams
    def make_bigrams(texts):
        return [bigram_model[doc] for doc in texts]
    data_words_bigrams = make_bigrams(data_words_nostops)
    # merge all to one list
    all_words = []
    for sentence in data_words_bigrams:
        all_words.extend(sentence)
    # count frequencies
    df = pd.DataFrame(all_words, columns = ['word'])
    df_counts = df['word'].value_counts()
    return df_counts


# testing count 2-grams 
print("count 2-grams ")
N = 20
tokens_generated_data_df = pd.DataFrame(np.zeros([len(generated_data_df), N]))
for i in range(len(generated_data_df)):
    print(" dimension {}:".format(str(i)))
    df_counts = most_frequent_words(generated_data_df.iloc[i,:].values, bigram_model)
    words = list(list(df_counts.index))
    print(words[:10])
    print("=====")
    tokens_generated_data_df.iloc[i,:] = words[:N] + [0]*(N - len(words))


# saving to file
tokens_generated_data_df.to_csv("results_words_generated.csv")


