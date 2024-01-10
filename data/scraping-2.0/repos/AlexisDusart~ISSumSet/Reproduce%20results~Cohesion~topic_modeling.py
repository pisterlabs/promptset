import pandas as pd
import os
from gensim import corpora, models
from gensim.models import LdaMulticore
import gensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import html
from tqdm import tqdm
import csv



"""
-----------------------------------------------------------------------------------------------------
Associate topics to tweets
-----------------------------------------------------------------------------------------------------
"""

for event in os.listdir("../../Sets/NHPAR/PreProcessed/"):
    if not event.startswith('.'):
        df = pd.read_csv('../../Sets/Tweets with ids/'+event,sep='\t',quoting=csv.QUOTE_NONE)
        df.columns = ["timestamp","id","uid","raw_text"]

        df["text"] = df["raw_text"].apply(lambda x: str(x))
        df["docs"] = df["text"].apply(lambda x: x.split())

        dictionary = corpora.Dictionary(df.docs)

        corpus = [dictionary.doc2bow(text) for text in df.docs]

        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, random_state=1, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

        l_document_topics = []
        l_id = list(df["id"])
        for i in corpus:
            l_document_topics.append(lda.get_document_topics(i))

        with open("../../Paper annotations/topic_modeling_annotations.txt",'a',encoding='UTF-8') as file:
            for i in range(len(l_id)):
                file.write(str(l_id[i])+'\t'+event[:-4]+'\t'+str(l_document_topics[i])+'\n')


"""
---------------------------------------------------------------------------------------------------------
Find optimal number of sub-event per topic
We do not report these score because there too much topics to annotate, we annotate with 3 topics for
each event
---------------------------------------------------------------------------------------------------------


df = pd.read_csv('../../Sets/Tweets with ids/albertaFloods2013.txt',sep='\t')
df.columns = ["timestamp","id","uid","raw_text"]

df["text"] = df["raw_text"].apply(lambda x: clean(x))
df["text"] = df["text"].apply(lambda x: stem(x))
df["docs"] = df["text"].apply(lambda x: x.split())

dictionary = corpora.Dictionary(df.docs)


corpus = [dictionary.doc2bow(text) for text in df.docs]
def compute_coherence_values(dictionary, texts, corpus, limit, start=2, step=3):

    from gensim.models.ldamodel import LdaModel
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, limit, step)):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=1, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, texts=list(df["docs"]), corpus=corpus, start=1, limit=20, step=1)

limit=20; start=1; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 2)) 

"""
"""
---------------------------------------------------------------------------------------------------------
lda, coherenceModel with maximised coherence c_v
- Exploring the Space of Topic Coherence Measures
if equals to 2 decimal places, choose the lowest number of topics
---------------------------------------------------------------------------------------------------------
"""
"""

map_number_topics_event = {
    'albertaFloods2013' : 7,
    'albertaWildfires2019' : 64,
    'australiaBushfire2013' : 13,
    'bostonBombings2013' : 40,
    'chileEarthquake2014' : 4,
    'coloradoStemShooting2019' : 11,
    'costaRicaEarthquake2012' : 1,
    'cycloneKenneth2019' : 59,
    'earthquakeBohol2013' : 1,
    'fireYMM2016' : 69,
    'flSchoolShooting2018' : 17,
    'guatemalaEarthquake2012' : 12,
    'hurricaneFlorence2018' : 28,
    'italyEarthquakes2012' : 7,
    'joplinTornado2011' : 7,
    'manilaFloods2013' : 4,
    'nepalEarthquake2015' : 120,
    'parisAttacks2015' : 33,
    'philipinnesFloods2012' : 1,
    'philippinesEarthquake2019' : 36,
    'queenslandFloods2013' : 74,
    'sandiegoSynagogueShooting2019' : 8,
    'shootingDallas2017' : 15,
    'southAfricaFloods2019' : 36,
    'typhoonHagupit2014' : 58,
    'typhoonYolanda2013' : 20,
}
"""