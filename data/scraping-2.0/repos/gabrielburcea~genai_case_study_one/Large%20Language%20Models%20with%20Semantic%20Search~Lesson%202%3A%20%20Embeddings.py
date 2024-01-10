# Databricks notebook source
"""Setup
Load needed API keys and relevant Python libaries."""

# COMMAND ----------

# !pip install cohere umap-learn altair datasets


# COMMAND ----------

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# COMMAND ----------

import cohere
co = cohere.Client(os.environ['COHERE_API_KEY'])

# COMMAND ----------

import pandas as pd

# COMMAND ----------

"""
Word Embeddings
Consider a very small dataset of three words.
"""

# COMMAND ----------

three_words = pd.DataFrame({'text':
  [
      'joy',
      'happiness',
      'potato'
  ]})

three_words

# COMMAND ----------

"""

Let's create the embeddings for the three words:
"""

# COMMAND ----------

three_words_emb = co.embed(texts=list(three_words['text']),
                           model='embed-english-v2.0').embeddings

# COMMAND ----------

word_1 = three_words_emb[0]
word_2 = three_words_emb[1]
word_3 = three_words_emb[2]

# COMMAND ----------

word_1[:10]

# COMMAND ----------

"""Sentence Embeddings
Consider a very small dataset of three sentences."""

# COMMAND ----------

sentences = pd.DataFrame({'text':
  [
   'Where is the world cup?',
   'The world cup is in Qatar',
   'What color is the sky?',
   'The sky is blue',
   'Where does the bear live?',
   'The bear lives in the the woods',
   'What is an apple?',
   'An apple is a fruit',
  ]})

sentences

# COMMAND ----------

"""Let's create the embeddings for the three sentences:"""

emb = co.embed(texts=list(sentences['text']),
               model='embed-english-v2.0').embeddings

# Explore the 10 first entries of the embeddings of the 3 sentences:
for e in emb:
    print(e[:3])

# COMMAND ----------

len(emb[0])

# COMMAND ----------

#import umap
#import altair as alt

# COMMAND ----------

from utils import umap_plot

# COMMAND ----------

chart = umap_plot(sentences, emb)

# COMMAND ----------

chart.interactive()

# COMMAND ----------

import pandas as pd
wiki_articles = pd.read_pickle('wikipedia.pkl')
wiki_articles

# COMMAND ----------

import numpy as np
from utils import umap_plot_big

# COMMAND ----------

articles = wiki_articles[['title', 'text']]
embeds = np.array([d for d in wiki_articles['emb']])

chart = umap_plot_big(articles, embeds)
chart.interactive()
