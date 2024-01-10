#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import tweepy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# In[3]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
def plotWordcloud(tweet_list):
    wordcloud = WordCloud(max_font_size=1000, max_words=500, background_color="white",normalize_plurals=False).generate(' '.join(tweet_list))
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    #plt.savefig("WordCloud_before"+".png", bbox_inches='tight')


# In[38]:


df_t= pd.read_csv("Data/labeled/Twitter_Sentiment.csv")
df_r = pd.read_csv("Data/labeled/Reddit_Sentiment.csv")
t=df_t['text']
r=df_r['body']
total=pd.concat([t, r], ignore_index=True)
total=total.to_frame()
total.count()
#plotWordcloud(total.array)


# In[42]:


total=total.dropna()
total.count()
total[0]


# In[43]:


import spacy
#spacy.download('en_core_web_sm')
#spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

en_stop = set(nltk.corpus.stopwords.words('english'))


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

import random
text_data = []
for line in total[0]:
    tokens = prepare_text_for_lda(line)
    text_data.append(tokens)


# In[140]:


import gensim

from gensim import corpora, models

dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

[[(dictionary[i], freq) for i, freq in doc] for doc in corpus[:1]]


# In[141]:


NUM_TOPICS = 125
#ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics = NUM_TOPICS, id2word=dictionary,
#                                           passes=50,minimum_probability=0.5,
#                                          per_word_topics=False)
#ldamodel.save('model5.gensim')
ldamodel = gensim.models.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary,
                                           passes=50,
                                          per_word_topics=False,
                                  alpha='auto')
from gensim.models import CoherenceModel
# Compute Perplexity
print('\nPerplexity: ', ldamodel.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=text_data, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[182]:


topics = ldamodel.show_topics(formatted=False, num_topics=NUM_TOPICS,num_words= 10)
for topic in topics:
    print(topic[0],[ t[0] for t in topic[1]])

#ldamodel.printTopics(NUM_TOPICS)
#import pandas as pd
#pd.DataFrame(ldamodel.get_document_topics(corpus))


# In[62]:


from gensim.models import CoherenceModel
# Compute Perplexity
print('Perplexity: ', ldamodel.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=text_data, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(' Coherence Score: ', coherence_lda,'\n')


# In[144]:


import pyLDAvis.gensim
import pyLDAvis

pyLDAvis.enable_notebook()
panel_genism = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, mds='tsne')
pyLDAvis.display(panel_genism)


# In[177]:


topics=[]
for i,c in enumerate(corpus):
    topic=ldamodel[c]
    topics.append(topic)
    
topics

topics_max=[]
topics_max2=[]
for i in topics:
    d=dict(i)
    topics_max.append(max(d, key=d.get))

topics_max
#corpus.shape


# In[178]:


t1=df_t[["created_at","text","label","score"]].copy()
t1['source']="twitter"
t1['created_at']=t1['created_at'].str.slice(stop=10)
t2=df_r[["timestamp","body","label","score"]].copy()
t2=t2.rename(columns={"timestamp":"created_at","body":"text"})
t2['source']="reddit"
t2['created_at']=t2['created_at'].str.slice(stop=10)
t=pd.concat([t1, t2], ignore_index=True)


# In[179]:


t.shape
t=t.dropna()
t.shape


# In[181]:


t['topic']=topics_max
t.to_csv("topics.csv")


# In[176]:





# In[92]:


from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer= CountVectorizer(input='content')
text_data_v =  [' '.join(text) for text in text_data]
dtm_c= count_vectorizer.fit_transform(text_data_v)


# In[95]:


ldamodel_sk = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=50, learning_method='online')
ldamodel_sk_result = ldamodel_sk.fit_transform(dtm_c)


# In[96]:


import pyLDAvis.sklearn
import pyLDAvis
pyLDAvis.enable_notebook() 
panel = pyLDAvis.sklearn.prepare(ldamodel_sk, dtm_c, count_vectorizer, mds='tsne')
pyLDAvis.display(panel)


# In[89]:


for n in range(ldamodel_sk_result.shape[0]):
    topic_most_pr = ldamodel_sk_result[n].argmax()
    print("doc: {} topic: {}\n".format(n,topic_most_pr))


# In[50]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer(input='content')
text_data_v =  [' '.join(text) for text in text_data]
dtm= vectorizer.fit_transform(text_data_v)


# In[8]:





# In[90]:


lda_model_2 = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=50, learning_method='online')
lda_model_2_result = lda_model_2.fit_transform(dtm)


# In[91]:


for n in range(lda_model_2_result.shape[0]):
    topic_most_pr = lda_model_2_result[n].argmax()
    print("doc: {} topic: {}\n".format(n,topic_most_pr))


# In[81]:


lda_model_2_result[0:0]


# ## import pyLDAvis.sklearn 
# import pyLDAvis
# ## conda install -c conda-forge pyldavis
# #pyLDAvis.enable_notebook() ## not using notebook
# pyLDAvis.enable_notebook()
# panel = pyLDAvis.sklearn.prepare(lda_model_2, dtm, vectorizer, mds='tsne')
# ### !!!!!!! Important - you must interrupt and close the kernet in Spyder to end
# ## In other words - press the red square and then close the small red x to close
# ## the Console
# pyLDAvis.display(panel)

# In[25]:


from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()] 

cloud = WordCloud(stopwords=STOPWORDS,
                  background_color='white',
                  width=3000,
                  height=2000,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = ldamodel.show_topics(formatted=False,num_topics=13)

fig, axes = plt.subplots(7, 2, figsize=(16,24), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# In[ ]:




