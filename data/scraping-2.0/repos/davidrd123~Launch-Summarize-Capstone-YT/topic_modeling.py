import os
from pathlib import Path
from gensim import corpora, models
from gensim.models import Phrases
from gensim.models.ldamodel import LdaModel
from gensim.models.phrases import Phraser
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
import numpy as np

import pyLDAvis 
import pyLDAvis.gensim_models as gensimvis

# from bertopic import BERTopic
import openai
# from bertopic.representation import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

nltk.download('stopwords')
nltk.download('wordnet')

# ALPHA = 0.61
# ETA = 0.91
# DECAY = 0.5

ALPHA = 0.31
ETA = 0.61
DECAY = 0.5

def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    short_word_free = " ".join(word for word in normalized.split() if len(word) > 2)
    return short_word_free

def compute_coherence_values_num_topics(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, alpha=ALPHA, eta=ETA, decay=DECAY)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def compute_coherence_values_params(dictionary, corpus, texts, alpha, eta, decay):
    coherence_values = []
    model_list = []
    for a in alpha:
        for e in eta:
            for d in decay:
                model = LdaModel(corpus=corpus, num_topics=9, id2word=dictionary, alpha=a, eta=e, decay=d)
                model_list.append(model)
                coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append((a, e, d, coherencemodel.get_coherence()))

    return model_list, coherence_values

def plot_coherence_values(start, limit, step, coherence_values):
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


list_of_transcripts = {}

transcripts_path = Path('transcripts')
# Iterate through each year
years = [year for year in os.listdir(transcripts_path) if year != '.DS_Store']
for year in years:
	# Get all directory names in the year directory not including .DS_Store
	project_names = [project for project in os.listdir(transcripts_path / year) if project != '.DS_Store']

	for project in project_names:
		# transcript is in the file matching '<video_id>_transcript.txt', regex match to filter the files down to it
		# transcript = [transcripts_path / year / project / file for file in os.listdir(transcripts_path / year / project) if file.endswith('_transcript.txt')][0]
		transcript = [transcripts_path / year / project / file for file in os.listdir(transcripts_path / year / project) if file.startswith('rewrite_gpt-3.5')][0]        
		with open(transcript) as f:
			list_of_transcripts[project] = f.read()

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

all_project_names = [name.lower() for name in list(list_of_transcripts.keys())]
additional_stopwords = ["like", "one", "also", "would", "need"]

# Add project names to the stop words
stop.update(all_project_names)
stop.update(additional_stopwords)
		
# Create a list of lists containing the tokens for each document in the corpus
texts = [clean(transcript).split() for transcript in list_of_transcripts.values()]

# Create bigram and trigram models
# bigram = Phrases(texts, min_count=5, threshold=100)  # higher threshold, fewer phrases.
# trigram = Phrases(bigram[texts], threshold=100)
# bigram_mod = Phraser(bigram)
# trigram_mod = Phraser(trigram)

# Apply the bigram and trigram models
# texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

# Create a dictionary representation of the documents.
dictionary = corpora.Dictionary(texts)

# Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) tuples.
corpus_bow = [dictionary.doc2bow(text) for text in texts]

texts_joined = [' '.join(text) for text in texts]
representation_model = OpenAI(model="gpt-3.5-turbo")
topic_model = BERTopic(embedding_model=representation_model, nr_topics=9, low_memory=True)

frequent_topics = topic_model.get_topic_freq()

# Train the LDA model
lda_model = models.LdaModel(corpus_bow, num_topics=8, id2word=dictionary, passes=15, alpha=ALPHA, eta=ETA, decay=DECAY)

lda_viz = gensimvis.prepare(lda_model, corpus_bow, dictionary)

# Print the topics
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)

corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] 
                 for topics in lda_model.get_document_topics(corpus_bow) ]

for project, corpus_topic in zip(list_of_transcripts.keys(), corpus_topics):
    print("Project ", project, " is about topic ", corpus_topic[0], " with a contribution of ", round(corpus_topic[1]*100, 2), "%")
    
# pyLDAvis.save_html(lda_viz, 'lda.html')




# coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v') 
# print('\nCoherence Score: ', coherence_model_lda.get_coherence())

### Compute Coherence Score - Num Topics
# start, limit, step = 2, 26, 1
# model_list, coherence_values = compute_coherence_values_num_topics(dictionary=dictionary, corpus=corpus_bow, texts=texts, start=start, limit=limit, step=step)
# plot_coherence_values(start, limit, step, coherence_values)

### Compute Coherence Score - Alpha / Eta / Decay
# alpha = list(np.arange(0.01, 1, 0.3))
# eta = list(np.arange(0.01, 1, 0.3))
# decay = list(np.arange(0.5, 1, 0.1))

# model_list, coherence_values = compute_coherence_values_params(dictionary=dictionary, corpus=corpus_bow, texts=texts, alpha=alpha, eta=eta, decay=decay)
# best_a, best_e, best_d, best_coherence = max(coherence_values, key=lambda item: item[-1])
# print("Best coherence score: ", best_coherence, " with alpha: ", best_a, " eta: ", best_e, " decay: ", best_d)
