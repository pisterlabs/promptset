import pandas as pd
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from bertopic import BERTopic

from hdbscan import HDBSCAN


data = pd.read_csv("arab_gen_twitter.csv")

data.head()

# shape  
data.shape
data = data.dropna()
documents = data['text'].values
arabert = TransformerDocumentEmbeddings('aubmindlab/bert-large-arabertv02-twitter')

# Topic ModelingL
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
topic_model = BERTopic(language="arabic", low_memory=True ,calculate_probabilities=False, embedding_model=arabert, hdbscan_model=hdbscan_model)

topics, probs = topic_model.fit_transform(documents)



#extract most frequent topics
topic_model.get_topic_freq().head(5)                     
topic_model.get_topic(1)[:10]

texts = [[word for word in str(document).split()] for document in documents]
id2word = corpora.Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]

topics=[]
for i in topic_model.get_topics():
  row=[]
  topic= topic_model.get_topic(i)
  for word in topic:
     row.append(word[0])
  topics.append(row)


# compute coherence score
#cm = CoherenceModel(topics=topics, texts=texts, corpus=corpus, dictionary=id2word, coherence='u_mass')
cm =  CoherenceModel(topics=topics, texts=texts, corpus=corpus, dictionary=id2word, coherence='c_v')
#cm = CoherenceModel(topics=topics, texts=texts, corpus=corpus, dictionary=id2word, coherence='u_mass')

coherence = cm.get_coherence() 
print('\nCoherence Score: ', coherence)


# Visualize the topics
#topic_model.visualize_topics()

# save the model 
topic_model.save("model_hdbscan")	



