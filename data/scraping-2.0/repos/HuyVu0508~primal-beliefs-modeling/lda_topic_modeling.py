import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy


# read file
df = pd.read_csv("tweets_NMF50.csv", header = 0)


# data cleaning 
data = df.message.values.tolist()
pprint(data[:10])


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# sentence to words
print("Sentence to words.")
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))
print(data_words[:1])


# remove the world is
print("Remove the world is.")
data_words = [sentence_tokens[3:] for sentence_tokens in data_words]
print(data_words[:10])


# Build the bigram and trigram models
print("Build the bigram and trigram models ")
bigram = gensim.models.Phrases(data_words, min_count=10, threshold=1) # higher threshold fewer phrases.
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)


# Define functions for stopwords, bigrams, trigrams and lemmatization
print("Define functions for stopwords, bigrams, trigrams and lemmatization")
def remove_stopwords(texts):
    stop_words = [ 'i', 'me', 'my', 'we', 'you', "you're", "you've", "you'll", "you'd", 'he', 'him', 'she', "she's",  'it', "it's", 'they', 'them', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        # texts_out.append([token.lemma_ if token.lemma_ not in ['-PRON-'] else token.text for token in doc])
    return texts_out


## Run methods in order
print("Run methods in order")
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# # python3 -m spacy download en
# nlp = spacy.load('en', disable=['parser', 'ner'])
# # Do lemmatization keeping only noun, adj, vb, adv
# data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# print(data_lemmatized[:1])
# no data lemmatization
data_lemmatized = data_words_bigrams
print(data_lemmatized[:10])


# Running CountVectorizer
print("Running CountVectorizer")
vectorizer = CountVectorizer(analyzer='word', min_df=10, lowercase=True, max_features=50000)     
# minimum occurences of a word 
# remove stop words
# convert all words to lowercase
# num chars > 3
# max number of uniq words
data_lemmatized = [" ".join(sentence) for sentence in data_lemmatized]
data_vectorized = vectorizer.fit_transform(data_lemmatized)


# Build LDA Model
print("Running LDA model")
lda_model = LatentDirichletAllocation(n_components=50, max_iter=10, learning_method='online', random_state=100, batch_size=128, evaluate_every = -1, n_jobs = -1)
# Max learning iterations
# Random state
# n docs in each learning iter
# compute perplexity every n iters, default: Don't
# Use all available CPUs
lda_output = lda_model.fit_transform(data_vectorized)
print(lda_model)  # Model attributes


# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))
# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))
# See model parameters
pprint(lda_model.get_params())


# Create Document — Topic Matrix
lda_output = lda_model.transform(data_vectorized)
# column names
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
# index names
docnames = ["Doc" + str(i) for i in range(len(data))]
# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic


# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda_model.components_)
# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
# View
df_topic_keywords.head()


# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)
# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
print("topic keywords: ")
print(df_topic_keywords.iloc[:,:10])


# saving to file
df_topic_keywords.to_csv("/data/hvu/bert-exploration/finetuned_model/gpt_wrapper_test_50/results_words_lda.csv", header = True, index = True)