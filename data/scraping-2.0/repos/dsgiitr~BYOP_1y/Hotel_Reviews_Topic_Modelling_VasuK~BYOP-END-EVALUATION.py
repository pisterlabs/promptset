import matplotlib.pyplot as plt # plotting the statistical distribution
import numpy as np # for linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

df = pd.read_csv("Restaurant_reviews_dataset.csv")

print(df)
print(len(df))

#For Displaying the first 5 rows of the DataFrame
print(df.head())

#For Displaying the shape of the DataFrame
print(df.shape)

#For Checking the data types of each column
print(df.dtypes)

#For Checking for missing values
print(df.isnull().sum())

#For Generating descriptive statistics for numerical columns
print(df.describe())

# Plotting a histogram of a numerical column
plt.hist(df['Liked'], bins=10)
plt.title('Likes & Dislikes visualisation')
plt.xlabel('Rating')
plt.ylabel('Number of reviews')
plt.show()

# Generating a boxplot of a numerical column
sns.boxplot(x=df['Liked'])
plt.title('Boxplot of Likes')
plt.xlabel('Column')
plt.show()

# Generating a correlation matrix of numerical columns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Creating a pairplot of numerical columns
sns.pairplot(df)
plt.title('Pairplot')
plt.show()

# Generating a bar chart of categorical columns
df['Liked'].value_counts().plot(kind='bar')
plt.title('Bar Chart of Column')
plt.xlabel('Column')
plt.ylabel('Frequency')
plt.show()

print(df.columns)
df = df.rename(columns={' Review':'Review'})

df['Review'] = df['Review'].str.replace("[^a-zA-Z#]", " ")
print(df)

da = df.Review.values.tolist()
print(da)

# Tokenizing the documents.
import nltk
from nltk.tokenize import RegexpTokenizer

# Splitting the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(da)):
    da[idx] = da[idx].lower()  # Convert to lowercase.
    da[idx] = tokenizer.tokenize(da[idx])  # Split into words.

# Removing numbers, but not words that contain numbers.
da = [[token for token in doc if not token.isnumeric()] for doc in da]

# Removing words that are only one character.
da = [[token for token in doc if len(token) >= 4] for doc in da]

# Lemmatizing the documents.
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
da = [[lemmatizer.lemmatize(token) for token in doc] for doc in da]
print(da)

# Computing bigrams.
from gensim.models import Phrases

# Adding bigrams and trigrams to df (only ones that appear 10 times or more).
bigram = Phrases(da, min_count=10)
for idx in range(len(da)):
    for token in bigram[da[idx]]:
        if '_' in token:
            # Token is a bigram, adding it to document.
            (da[idx].append(token))

# Removing rare and common tokens.
from gensim.corpora import Dictionary

# Creating a dictionary representation of the documents.
dictionary = Dictionary(da)

# Filtering out words that occur less than 5 documents, or more than 20% of the documents.
dictionary.filter_extremes(no_below=5, no_above=0.2)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in da]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

# Training the LDA model.
from gensim.models import LdaModel

# Setting the training parameters.
num_topics = 11
chunksize = 1000
passes = 20
iterations = 400
eval_every = None # I have kept it to none because calculating model perplexity takes too much memory and time.
# Making an index to word dictionary.
# loading dictionary
temp = dictionary[0]
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha=0.01,
    eta=0.9,
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

print(model)

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

from IPython.core.getipython import get_ipython
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
print(vis)

import gensim.downloader as api
from gensim.corpora import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
da = api.load('text8')

dictionary = Dictionary()
for doc in da:
    dictionary.add_documents([[lemmatizer.lemmatize(token) for token in doc]])
dictionary.filter_extremes(no_below=20, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in da]

from gensim.models import LdaModel
topic_model_class = LdaModel
ensemble_workers = 4
num_models = 8
num_topics = 20
passes = 5

from gensim.models import EnsembleLda

ensemble = EnsembleLda(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    passes=passes,
    num_models=num_models,
    topic_model_class=LdaModel,
    ensemble_workers=ensemble_workers,

)

print(len(ensemble.ttda))
print(len(ensemble.get_topics()))

from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

# Computing coherence score using c_v measure
coherence_model_lda = CoherenceModel(model=model, texts=da, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

print("Coherence Score:", coherence_lda)

import numpy as np
shape = ensemble.asymmetric_distance_matrix.shape
without_diagonal = ensemble.asymmetric_distance_matrix[~np.eye(shape[0], dtype=bool)].reshape(shape[0], -1)
print(without_diagonal.min(), without_diagonal.mean(), without_diagonal.max())

ensemble.recluster(eps=0.09, min_samples=2, min_cores=2)

print(len(ensemble.get_topics()))
