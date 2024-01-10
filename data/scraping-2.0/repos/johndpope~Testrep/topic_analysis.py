import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import sqlite3
import math
import nltk
from nltk.stem.snowball import SnowballStemmer
from datetime import datetime

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def freq(text,x):
    freq = pd.Series(''.join(str(text)).split()).value_counts()[:x]
    return list(freq.index)

def create_bigram(data_words):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    # trigram_mod = gensim.models.phrases.Phraser(trigram)
    return gensim.models.phrases.Phraser(bigram)


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def make_bigrams(text):
    return [bigram_mod[doc] for doc in text]

def assign_weight(doc_complete,weight_by):
    # assign weight by duplicate text
    if weight_by != 'no':
        k = 0
        while k < len(cleaned_tweets_en):
            tweet_id = int(cleaned_tweets_en.id[k])
            user_id = int(cleaned_tweets_en.user_id[k])
            count = int(user_table.loc[user_table.id == user_id, weight_by]) + 0.1
            i = 0
            while i < round(math.log(count)):
                doc_complete.append(doc_complete[k])
                i = i + 1
            # print('finish weight', k)
            k = k + 1
        print('Weight successfully')
        return doc_complete
    else:
        return doc_complete

def save_noun(doc_clean):
    new_doc =[]
    for doc in doc_clean:
        tagged = nltk.pos_tag(doc)
        for (word, tag) in tagged:
            if not tag.startswith("N") or tag == "FW":
                doc.remove(word)
            else:
                continue
        new_doc.append(doc)
    return new_doc

def stem(doc_clean) :
    new_doc = []
    for doc in doc_clean:
        new_word = []
        for word in doc:
            word = SnowballStemmer('english').stem(word)
            new_word.append(word)
        new_doc.append(new_word)
    return new_doc


# Path to database
database_file_name = 'C:\\Users\Poramapa\PycharmProjects\scrapelimburg\data\db_tweets_27Jan.sqlite'
# Read data from db file
conn = sqlite3.connect(database_file_name)
cursor = conn.cursor()
print("Opened database successfully")

# read tables
# tweets_table = pd.read_sql_query("SELECT  id, is_truncated, user_id, lang, retweet_count, is_retweeted, bounding_box_id  "
#                                  "FROM tweets WHERE id IN (SELECT id FROM cleaned_tweets)", conn)
user_table = pd.read_sql_query("SELECT  * FROM users", conn)
cleaned_tweets_en = pd.read_sql_query("select distinct user_mention.screen_name, user_mention.user_mention_id, group_concat(cleaned_tweets.stem_tweets, ' ') "
                                      "from  user_mention inner join cleaned_tweets on cleaned_tweets.id = user_mention.tweet_id where"
                                      "(cleaned_tweets.stem_tweets is not Null and cleaned_tweets.clean_retweets is Null "
                                      "and cleaned_tweets.stem_tweets <> '' and cleaned_tweets.id "
                                      "in(select id from tweets where (lang = 'en'))) group by user_mention.user_mention_id", conn)
cleaned_tweets_en = cleaned_tweets_en.rename(columns={"group_concat(cleaned_tweets.stem_tweets, ' ')": "stem_tweets"})
cleaned_tweets_en.stem_tweets = cleaned_tweets_en.stem_tweets.apply(lambda x: " ".join(x for x in str(x).split() if x != "http"))

freq_tweet = freq(cleaned_tweets_en.stem_tweets, 20)
print(freq_tweet)
cleaned_tweets_en['stem_tweets_nofreq'] = cleaned_tweets_en.stem_tweets.apply(lambda x: " ".join(x for x in str(x).split() if x not in freq_tweet))
cleaned_tweets_en = cleaned_tweets_en.drop(cleaned_tweets_en[cleaned_tweets_en.stem_tweets_nofreq.str.split().str.len() < 3].index)

    #
    # ("select created_at, cleaned_tweets.id, clean_tweets, lang, stem_tweets, retweet_count, favorite_count,is_retweeted,"
    #                                   "bounding_box_id, user_id from cleaned_tweets inner join tweets on cleaned_tweets.id"
    #                                 "= tweets.id where (tweets.lang = 'en' and cleaned_tweets.stem_tweets is not NULL)", conn)


# strip frequent words
# freq_tweet = freq(cleaned_tweets_en.stem_tweets)
# print(freq_tweet)
# cleaned_tweets_en.stem_tweets_nofreq = cleaned_tweets_en.stem_tweets(lambda x: " ".join(x for x in str(x).split() if x not in freq_tweet))


# cleaned_tweets_en['created_at'] = pd.to_datetime(cleaned_tweets_en['created_at'])


# prepare text for analysis
# tweets_dec = cleaned_tweets_en[cleaned_tweets_en['created_at'].dt.month == 12]
# tweets_jan = cleaned_tweets_en[cleaned_tweets_en['created_at'].dt.month == 1]
data_words = list(sent_to_words(cleaned_tweets_en.stem_tweets_nofreq))
print(data_words)

#group 2 words that normally appear togather ex) a person name : javad_zarif
bigram_mod = create_bigram(data_words)
data_words_bigrams = make_bigrams(data_words)

#save only noun
doc_noun = save_noun(data_words_bigrams)
# doc_noun = stem(doc_noun)



# Assign weight
# insert 'no' if you don't want to weight
doc_clean = assign_weight(doc_noun,'no')

# Creating the term dictionary of our corpus, where every unique term is assigned an index.
print('Create Dictionary')
id2word = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
print('Converting list of documents')
corpus = [id2word.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
print('Create object for LDA')
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
print('Train LDA')
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=8, passes=20)

print(lda_model.print_topics())
doc_lda = lda_model[corpus]


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_clean, corpus=corpus,dictionary=id2word, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

 # Visualize the topics and save in html file
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization_user_mention_27Jan.html')
