import pandas as pd
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from umap import UMAP


  
# Dataset from ArabGend: Gender analysis and inference on Arabic Twitter
data = pd.read_csv("arab_gen_twitter.csv")

data.head()

# shape  
data.shape
data = data.dropna()
documents = data['text'].values
arabert = TransformerDocumentEmbeddings('aubmindlab/bert-base-arabertv02')

# Topic Modeling
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
topic_model = BERTopic(language="arabic", low_memory=True ,calculate_probabilities=False, embedding_model=arabert, umap_model=umap_model)
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
cm = CoherenceModel(topics=topics, texts=texts, corpus=corpus, dictionary=id2word, coherence='c_npmi')
coherence = cm.get_coherence() 
print('\nCoherence Score: ', coherence)


# Visualize the topics
#topic_model.visualize_topics()
