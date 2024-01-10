# --------------
# import libraries
import numpy as np
import pandas as pd
import re

# Load data
data = pd.read_csv(path, parse_dates = [0], infer_datetime_format = True)

# Sort headlines by date of publish
data.sort_values('publish_date', inplace = True)

# Retain only alphabets
data['headline_text'] = data['headline_text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

# Look at the shape of data
print(data.shape)

# Look at the first first five observations
print(data.head(5))


# --------------
# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words = 'english', max_features = 30000)

# Transform headlines
news = vectorizer.fit_transform(data['headline_text'])

# initialize empty dictionary
words = {}

# initialize with 0
i = 0

# Number of time every feature appears over the entire document
sums = np.array(np.sum(news, axis=0)).flatten()
print(sums)
# Loop to map 'sums' to its word
for word in vectorizer.get_feature_names():
    words[word] = sums[i]
    i += 1

# Top 20 most occuring words
top_20 = sorted(words.items(), key = operator.itemgetter(1), reverse = True)[:20]

# Display top 20 words
top_20_words = [i[0] for i in top_20]
top_20_values = [i[1] for i in top_20]

print(top_20_words)


# --------------
# import libraries
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import pprint

# number of topics
n_topics = 5

# initialize SVD 
lsa_model = TruncatedSVD(n_components = n_topics, random_state = 2)

# fit and transform 'news' 
lsa_topic_matrix = lsa_model.fit_transform(news)
print(lsa_topic_matrix.shape)

'''We are not interested in knowing every word of a topic.
Instead, we want to look at the first (lets say) 10 words
of a topic'''

# empty dictionary to store topic number and top 10 words for every topic 
topic_lsa = {}

# loop over every topic
for i, topic in enumerate(lsa_model.components_):
    topic_lsa["Topic {}".format(i)] = ' + '.join([(vectorizer.get_feature_names()[j] + '*' + str(topic[j])) for j in topic.argsort()[:-11:-1]])

# pretty print topics
pprint.pprint(topic_lsa)



# --------------
# import libraries
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from nltk import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import matplotlib.pyplot as plt

# stopwords list
stop = set(stopwords.words('english'))

# string punctuations 
exclude = set(string.punctuation)

# lemmatizer
lemma = WordNetLemmatizer()

# Function to clean data from stopwords, punctuation marks and lemmatize
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


# Code starts here


# convert headlines to list
headlines = data['headline_text'].tolist()
print(headlines[9])
# cleaned data
clean_headlines = [clean(row).split() for row in headlines]
print(clean_headlines[9])
# Creating the term dictionary of our courpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(clean_headlines)
print(dictionary)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_headlines]
#print(doc_term_matrix)
# build LDA model
lda_model = LdaModel(doc_term_matrix, num_topics = 5, id2word = dictionary, random_state = 2, iterations = 10)

# extract topics for headlines
topics = lda_model.print_topics(num_topics = 5, num_words = 10)

# pprint topics
pprint.pprint(topics)

# Code ends here


# --------------
# coherence score



# Function to calculate coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """

    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics = num_topics, random_state = 2, id2word = dictionary, iterations = 10)
        model_list.append(model)

        coherence_model = CoherenceModel(model = model ,texts = texts, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherence_model.get_coherence())

    return (model_list, coherence_values)

# Can take a long time to run
coherence_model_lda = CoherenceModel(model = lda_model, texts = clean_headlines, dictionary = dictionary, coherence = 'c_v')

coherence_lda = coherence_model_lda.get_coherence()
print(coherence_lda)

model_list, coherence_values = compute_coherence_values(dictionary = dictionary, corpus = doc_term_matrix, texts = clean_headlines, limit = 50, start=2, step=6)

x = range(2, 50, 6)

# Plotting
plt.figure()
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.show()



