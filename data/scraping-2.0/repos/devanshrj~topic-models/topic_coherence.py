from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

import argparse
import pandas as pd
import sqlalchemy
import warnings

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("--database", default="Reddit_Depression_and_India", type=str)
parser.add_argument("--msgs", default="msgs_posts", type=str)
parser.add_argument("--topics", default="topwords_posts_200", type=str)
args = parser.parse_args()
print(args)


db = sqlalchemy.engine.url.URL(drivername='mysql',
                               host='127.0.0.1',
                               database=args.database,
                               query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})


engine = sqlalchemy.create_engine(db)


print("--- Reading corpus ---")
corpus_query = f'''SELECT * from {args.msgs};'''
corpus_df = pd.read_sql(corpus_query, engine)
print(corpus_df.head())
messages_li = corpus_df['message'].tolist()

print("--- Tokenizing corpus ---")
texts = []
for msg in messages_li:
    if msg is None:
       msg = ''
    texts.append(msg.split())
print("--- Tokenized corpus! ---")


print("--- Reading topics ---")
topics_query = f'''SELECT * from {args.topics};'''
topics_df = pd.read_sql(topics_query, engine)
print(topics_df.head())
topics_li = topics_df['termy'].tolist()

print("--- Tokenizing topics ---")
topics = []
for topic in topics_li:
    topics.append(topic.split(', '))
print("--- Tokenized topics! ---")
# print(topics)
print(len(topics))


print("--- Creating dictionary ---")
dictionary = Dictionary(texts)
print("--- Dictionary created! ---")


print("--- Calculating coherence scores ---")
cm1 = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='u_mass')
coherence1 = cm1.get_coherence()
print("u_mass:" ,coherence1)

# cm2 = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
# coherence2 = cm2.get_coherence()
# print('c_v:',coherence2)

# cm3 = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_uci')
# coherence3 = cm3.get_coherence()
# print('c_uci:',coherence3)

# cm4 = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_npmi')
# coherence4 = cm4.get_coherence()
# print('c_npmi:',coherence4)