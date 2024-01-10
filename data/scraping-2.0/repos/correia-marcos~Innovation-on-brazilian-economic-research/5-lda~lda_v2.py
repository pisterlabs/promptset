"""Script for applying a Lemma instead of stemmer in this project.

We followed the research of the best lemmatizers projects on python, and Stanza
is one of the best nowdays. Please check the following link for more
information:
https://universaldependencies.org/conll18/results-lemmas.html

We first tried the TurkuNLP, but downloading and using the library is just
to hard for the inexperienced programers envolved in this project. For the
facility of using Stanza and the quality of results, Stanza is arguably the
best library for our purpose (the difference of results between TurkuNLP and
                              Stanza is actually quiet low).

VERY IMPORTANT: In this script we did NOT do Tfidf-Vectorizer!!!!
"""
import os
import pandas as pd
import pickle
import logging
from gensim import matutils, corpora
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import gensim
from nltk import word_tokenize
import pyLDAvis
# import pyLDAvis.gensim
import matplotlib.pyplot as plt
%matplotlib inline

root = os.path.abspath(os.curdir)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

local = r'/dados/data_final'
corpus_en = pd.read_pickle(f'{root}/{local}/corpus_cleaned4_en.pkl')
corpus_pt = pd.read_pickle(f'{root}/{local}/corpus_cleaned4_pt.pkl')


stopwords = pickle.load(open(f'{root}/{local}\\stopwords_total.pkl', 'rb'))
# stopwords.append('-')
# stopwords.append('economy')
# stopwords.append('economia')
# stopwords.append('alpha')
# stopwords.append('beta')
# stopwords.append('gamma')
# stopwords.append('●●●')
# stopwords.append('∗∗∗')
# Did some removing of "cidcidcid..." terms by spyder IDE operations. It was
# easier then "stopwords.remove('...')".

# pickle.dump(stopwords, open(f'{root}/{local}\\stopwords_total.pkl', 'wb'))


def prepare_text_for_lda(text):
    """Do the tokenization removind stopwords."""
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [token for token in tokens if len(token) > 2]

    return tokens


corpus_en['tokenized'] = corpus_en['texts'].apply(prepare_text_for_lda)
corpus_pt['tokenized'] = corpus_pt['texts'].apply(prepare_text_for_lda)


# =============================================================================
# Trying the Gensim full coding LDA - ENGLISH
# =============================================================================

id2word_en_gensim = corpora.Dictionary(corpus_en.tokenized)
id2word_en_gensim.filter_extremes(no_below=2)

len(id2word_en_gensim)

corpus_en_gensim = [id2word_en_gensim.doc2bow(text) for text in
                    corpus_en.tokenized]

num_topics = 30
lda_en = LdaModel(corpus=corpus_en_gensim, id2word=id2word_en_gensim,
                  num_topics=num_topics, passes=60,
                  random_state=42, iterations=200,
                  alpha='auto', chunksize=1100,
                  minimum_probability=0.00)

topicos_en = lda_en.get_document_topics(corpus_en_gensim,
                                        minimum_probability=0.0)

docs_topic_en = pd.DataFrame()
dict_en = {}
indexes = []
for doc in range(len(topicos_en)):
    print(doc)
    index_docs = corpus_en.index.array[doc]
    indexes.append(index_docs)
    for i in topicos_en[doc]:
        index_topico, valor = i
        dict_en[index_topico] = valor
    df = pd.DataFrame(dict_en, index=indexes)
    docs_topic_en = docs_topic_en.append(df)
    indexes = []
    dict_en = {}


docs_topic_en.to_pickle(f'{root}/{local}/docs_topic_en({num_topics})v3.pkl')
# =============================================================================
# Trying the Gensim full coding LDA - PORTUGUESE
# =============================================================================

id2word_pt_gensim = corpora.Dictionary(corpus_pt.tokenized)
id2word_pt_gensim.filter_extremes(no_below=5)

len(id2word_pt_gensim)

corpus_pt_gensim = [id2word_pt_gensim.doc2bow(text) for text in
                    corpus_pt.tokenized]

num_topics = 30
lda_pt = LdaModel(corpus=corpus_pt_gensim, id2word=id2word_pt_gensim,
                  num_topics=num_topics, passes=60,
                  random_state=42, iterations=200,
                  alpha='auto', chunksize=1100,
                  minimum_probability=0.00)

topicos_pt = lda_pt.get_document_topics(corpus_pt_gensim,
                                        minimum_probability=0.0)

docs_topic_pt = pd.DataFrame()
dict_pt = {}
indexes = []
for doc in range(len(topicos_pt)):
    print(doc)
    index_docs = corpus_pt.index.array[doc]
    indexes.append(index_docs)
    for i in topicos_pt[doc]:
        index_topico, valor = i
        dict_pt[index_topico] = valor
    df = pd.DataFrame(dict_pt, index=indexes)
    docs_topic_pt = docs_topic_pt.append(df)
    indexes = []
    dict_en = {}

docs_topic_pt.to_pickle(fr'{root}/{local}\docs_topic_pt({num_topics})v3.pkl')

# =============================================================================
# Quality test
# =============================================================================

coerencia_model_pt = CoherenceModel(model=lda_pt,
                                    texts=corpus_pt.tokenized,
                                    dictionary=id2word_pt_gensim,
                                    coherence='c_v')
coerencia_pt = coerencia_model_pt.get_coherence()

eita = lda_en.print_topics()


coerencia_model_en = CoherenceModel(model=lda_en,
                                    texts=corpus_en.tokenized,
                                    dictionary=id2word_en_gensim,
                                    coherence='c_v')
coerencia_en = coerencia_model_en.get_coherence()


for doc in range(len(topicos_pt)):
    print(lda_pt[corpus_sp_pt[doc]])


# =============================================================================
# Visualizations
# =============================================================================
pyLDAvis.enable_notebook()
vis_en = pyLDAvis.gensim.prepare(lda_en, corpus_en_gensim, id2word_en_gensim)
pyLDAvis.show(vis_en)

vis_pt = pyLDAvis.gensim.prepare(lda_pt, corpus_pt_gensim, id2word_pt_gensim)
pyLDAvis.show(vis_pt)

os.mkdir(f'{root}/{local}/pyLDAvis')

pyLDAvis.save_html(vis_en, f'{root}/{local}/lda_en.html')
pyLDAvis.save_html(vis_pt, f'{root}/{local}/lda_pt.html')
