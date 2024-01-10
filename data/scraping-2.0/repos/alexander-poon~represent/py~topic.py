#!/usr/bin/env python

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel

# TODO: Ensure dependency scripts are run before loading
print('Loading data...')

legacy = pd.read_json('data/legacy.json')
current = pd.read_json('data/current.json')
tokens = pd.read_json('data/tokens.json', typ='series')

bills = pd.concat([legacy, current], ignore_index=True, sort=False)

# Pick a subject and number of topics
subject = 'Education'
n_topics = 20

# Extract bills and preprocessed tokens corresponding to selected topic
index = bills['subjects'].apply(lambda x: '|'.join(x)).str.contains(subject)
tokens = tokens[index].reset_index(drop=True)
bills = bills[index].reset_index(drop=True)

# Run Topic Model
print('Running topic model for', subject, 'bills...')

dictionary = Dictionary(tokens)
corpus = [dictionary.doc2bow(b) for b in tokens]

lda_model = LdaModel(
	corpus=corpus,
	id2word=dictionary,
	num_topics=n_topics,
	passes=50,
	alpha='auto',
	eta='auto',
	random_state=79
)

# Evaluate Topic Model
c = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')

print('Model Coherence: ', c.get_coherence())

# How much does each bill fall under each topic?
def get_bill_topics(model):
	topics = pd.DataFrame()

	for i, row in enumerate(model[corpus]):
		# Proportion that document falls into each topic
		topics = pd.concat([topics, pd.DataFrame(model[corpus][i]).set_index(0)], axis=1)

	topics = topics.transpose().reset_index(drop=True)

	# Integer index of dominant topic
	dominant_topic = topics.idxmax(axis=1).rename('dominant_topic')

	# Percentage that document represents dominant topic
	max_perc = topics.max(axis=1, skipna=True).rename('max_perc')

	return pd.concat([bills[['session', 'bill_id', 'title', 'text']], dominant_topic, max_perc, topics], axis=1)


bill_topics = get_bill_topics(lda_model)

print('Exporting topic model...')
lda_model.save('models/topic_model_' + subject.lower())

print('Done!')
