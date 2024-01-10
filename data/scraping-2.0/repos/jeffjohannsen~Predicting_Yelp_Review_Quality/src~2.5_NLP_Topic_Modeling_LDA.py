import numpy as np
import pandas as pd
import pickle as pkl
from pprint import pprint
# Connecting to Postgres RDS on AWS
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# visuals
# import pyLDAvis
# import pyLDAvis.gensim
# gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.models import CoherenceModel
# nltk
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
# spacy
import spacy

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)

db_endpoint = None
db_password = None

engine = create_engine(
            f"postgresql+psycopg2://postgres:{db_password}@{db_endpoint}/yelp_2021_db"
            )

train = pd.read_sql(sql=f"SELECT review_id, review_text FROM text_data_train", con=engine)
test = pd.read_sql(sql=f"SELECT review_id, review_text FROM text_data_test", con=engine)

print("Data Loaded")

# Stopwords
stop_words = stopwords.words('english')
stop_words.extend([])

# Spacy Prep Model
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def preprocess_text(df, stopwords):
    # Convert to List
    text = df['review_text'].values.tolist()
    # Create Tokens
    text_list = list(map(lambda x: (gensim.utils.simple_preprocess(str(x), deacc=True)), text))
    # Remove Stopwords
    text_list = list(map(lambda x: [word for word in x if word not in stopwords], text_list))
    # Add Bigrams
    bigram = gensim.models.Phrases(text_list, min_count=5, threshold=50)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    text_list = list(map(lambda x: bigram_mod[x], text_list))
    # Lemmatize
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    tokens = list(map(lambda x: [token.lemma_ for token in nlp(" ".join(x)) if token.pos_ in allowed_postags], text_list))
    return tokens

# processed_train = preprocess_text(train, stop_words)
# processed_test = preprocess_text(test, stop_words)

# print("Preprocess Complete")

# f = open("processed_train_lda.pkl", "wb")
# pkl.dump(processed_train, f)
# f.close()

# f = open("processed_test_lda.pkl", "wb")
# pkl.dump(processed_test, f)
# f.close()

# print("Preprocessed Text Saved")

processed_train = None
processed_test = None

with open("processed_train_lda.pkl", "rb") as input_file:
    processed_train = pkl.load(input_file)

with open("processed_test_lda.pkl", "rb") as input_file:
    processed_test = pkl.load(input_file)

print("Processed Text Loaded")

# Prepare Data for Model
id2word = corpora.Dictionary(processed_train)
train_corpus = [id2word.doc2bow(text) for text in processed_train]
test_corpus = [id2word.doc2bow(text) for text in processed_test]

print("Corpus Created")

# def eval_lda_models(bow_corpus, id2word, processed_texts, topic_counts_to_test):
#     results = {}
#     lda = None
#     for i in topic_counts_to_test:
#         lda = gensim.models.LdaMulticore(bow_corpus, num_topics=i, id2word=id2word, passes=5)
#         perplexity = lda.log_perplexity(bow_corpus)
#         coherence_model_lda = CoherenceModel(model=lda, texts=processed_texts, dictionary=id2word, coherence='c_v')
#         coherence = coherence_model_lda.get_coherence()
#         results[f'{i}_topics'] = {}
#         results[f'{i}_topics']['model'] = lda
#         results[f'{i}_topics']['perplexity'] = perplexity
#         results[f'{i}_topics']['coherence'] = coherence
#         print(f'{i}_topics: {results[f"{i}_topics"]}')
#     return results

# lda_eval_results = eval_lda_models(train_corpus, id2word, processed_train, [3, 5, 7, 10, 20, 50, 100])

# Train Model
corpus_for_training = train_corpus[:1000000]
print(f'Corpus for Training Shape: {len(corpus_for_training)}')

final_lda_model = gensim.models.LdaMulticore(corpus_for_training, id2word=id2word, num_topics=5)
print("Model Training Complete")

# Save Model
final_lda_model.save("LDA_Model_1M")
print("Model Saved")

# Load Model
# final_lda_model = gensim.models.LdaMulticore.load('LDA_Model_1M', mmap='r')
# print("Model Loaded")

# Create Feature Vectors
topic_count = 5

train_dicts = []
for i in range(len(train)):
    topics = final_lda_model.get_document_topics(train_corpus[i], minimum_probability=0.0)
    topic_dict = {f'topic_{j}_lda':topics[j][1] for j in range(topic_count)}
    train_dicts.append(topic_dict)

finished_train = pd.concat([train, pd.DataFrame(train_dicts)], axis=1)
finished_train = finished_train.drop(columns=['review_text'])

print("Train Data Complete")

test_dicts = []
for i in range(len(test)):
    topics = final_lda_model.get_document_topics(test_corpus[i], minimum_probability=0.0)
    topic_dict = {f'topic_{j}_lda':topics[j][1] for j in range(topic_count)}
    test_dicts.append(topic_dict)

finished_test = pd.concat([test, pd.DataFrame(test_dicts)], axis=1)
finished_test = finished_test.drop(columns=['review_text'])

print("Test Data Complete")

# Save Data
finished_train.to_sql(
        "text_lda_train",
        con=engine,
        index=False,
        if_exists="replace",
    )

finished_test.to_sql(
        "text_lda_test",
        con=engine,
        index=False,
        if_exists="replace",
    )

print("Save to RDS Complete")
print("Done")
print("--------------------------------")
