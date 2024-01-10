# %%
from bertopic import BERTopic
import pandas as pd
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# %%
nltk.download('punkt')
nltk.download('stopwords')

# %%
stopwords = list(stopwords.words('norwegian'))

# %% Load data

train_df = pd.read_csv('../data/training_and_test_data/output_enc_concat_train.csv')
test_df = pd.read_csv('../data/training_and_test_data/output_enc_concat_test.csv')

texts_train = train_df['text']
texts_test = test_df['text']

# %%

representation_model = KeyBERTInspired()
vectorizer_model = CountVectorizer(stop_words=stopwords)
# umap_model = UMAP(n_neighbors=15, n_components=10, metric='cosine', low_memory=False)
# hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')

# For coherence calculation (NPMI)
texts_tokenized = [text.split() for text in texts_train]
dictionary = Dictionary(texts_tokenized)

nr_topics_range = [5, 6, 7, 8, 9, 10, 15, 20]
results = []

for nr_topics in nr_topics_range:
    topic_model = BERTopic(language="multilingual", min_topic_size=20, nr_topics=nr_topics,
                           representation_model=representation_model, vectorizer_model=vectorizer_model)
    topics_train, probs_train = topic_model.fit_transform(texts_train)

    print(f"topic info for number of topics: {nr_topics}")
    fig = topic_model.visualize_barchart(title=f'Topic Word Scores for nr of topics = {nr_topics}', top_n_topics=8,
                                         n_words=7, width=300, height=300)
    fig.show()

    # cross tab
    topics_test, probs_test = topic_model.transform(texts_test)
    test_df['Predicted_BERTopics'] = topics_test
    pd.set_option('display.max_rows', None)
    cross_tab_test = pd.crosstab(test_df['label'], test_df['Predicted_BERTopics'])
    cross_tab_test.to_csv(f'berttopic_real_label_{nr_topics}.csv', index=True)

    extracted_topics = topic_model.get_topics()
    valid_topic_ids = [topic_id for topic_id in extracted_topics if topic_id >= 0]
    topic_words = [[word for word, _ in topic_model.get_topic(topic_id)] for topic_id in extracted_topics]

    # Calculate NPMI
    coherence_model = CoherenceModel(topics=topic_words, texts=texts_tokenized, dictionary=dictionary,
                                     coherence='c_npmi')
    coherence_score = coherence_model.get_coherence()

    results.append((nr_topics, coherence_score))

for nr_topics, score in results:
    print(f"NPMI Coherence Score for {nr_topics} topics: {score}")

