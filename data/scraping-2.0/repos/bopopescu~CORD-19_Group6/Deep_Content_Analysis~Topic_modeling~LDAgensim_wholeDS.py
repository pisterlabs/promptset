# Apply GENSIM

import pandas as pd
import csv
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt


# open metadata file and cleaning
with open(r"E:\Helen\FinalProject_INFO5731\CSV_formatted\DS_dividedbyYears\DS1\after2020_DS1.csv", "r",  newline="", encoding='utf-8') as file:
    df1= pd.read_csv(file)

file.close()

df1['abstract']=df1['abstract'].apply(lambda x: " ".join(x for x in str(x).split() if not x.isdigit() and not x.isspace()))
df1['abstract']=df1['abstract'].str.replace('[^\w\s,]','')
#df1['abstract']=df1['abstract'].str.lower()

with open(r"E:\Helen\FinalProject_INFO5731\CSV_formatted\DS_dividedbyYears\DSApr3\after2020_newOnly_Ap3.csv", "r",  newline="", encoding='utf-8') as file:
    df2= pd.read_csv(file)

file.close()

df2['abstract']=df2['abstract'].apply(lambda x: " ".join(x for x in str(x).split() if not x.isdigit() and not x.isspace()))
df2['abstract']=df2['abstract'].str.replace('[^\w\s,]','')
#df2['abstract']=df2['abstract'].str.lower()

with open(r"E:\Helen\FinalProject_INFO5731\CSV_formatted\DS_dividedbyYears\DSApr10\after2020_newOnly_Apr10.csv", "r",  newline="", encoding='utf-8') as file:
    df3= pd.read_csv(file)

file.close()

df3['abstract']=df3['abstract'].apply(lambda x: " ".join(x for x in str(x).split() if not x.isdigit() and not x.isspace()))
df3['abstract']=df3['abstract'].str.replace('[^\w\s,]','')
#df3['abstract']=df3['abstract'].str.lower()

with open(r"E:\Helen\FinalProject_INFO5731\CSV_formatted\DS_dividedbyYears\DSApr17\after2020_newOnly_Apr17.csv", "r",  newline="", encoding='utf-8') as file:
    df4= pd.read_csv(file)

file.close()

df4['abstract']=df4['abstract'].apply(lambda x: " ".join(x for x in str(x).split() if not x.isdigit() and not x.isspace()))
df4['abstract']=df4['abstract'].str.replace('[^\w\s,]','')
#df4['abstract']=df4['abstract'].str.lower()


# Topic modeling with LDA and Gensim
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')
stop_plus = ['word', 'count', 'text', 'all', 'right', 'no', 'without', 'abstract', 'no', 'reuse', 'without', 'abstract', 'nan']

# Create PorterStemmer
p_stemmer = PorterStemmer()
# create list of documents
abstract_set = []
for abstract in df1['abstract'].dropna():
    abstract_set.append(abstract)

for abstract in df2['abstract'].dropna():
    abstract_set.append(abstract)

for abstract in df3['abstract'].dropna():
    abstract_set.append(abstract)

for abstract in df4['abstract'].dropna():
    abstract_set.append(abstract)

# list for tokenized documents in loop
texts = []

# loop through document list
for i in abstract_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop + stop_plus]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens if len(i)>3]
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
#print(dictionary)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
#print(corpus

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=30, id2word = dictionary, passes=60)

topics = []
for topic in ldamodel.print_topics(num_topics=30,num_words=12):
    topics.append(topic)
print(topics)


df_topics = pd.DataFrame(topics)

with open (r"E:\Helen\FinalProject_INFO5731\ALL_OUTPUTS\LDAgensim_wholeDS.csv", 'w',  newline="",
         encoding='utf-8') as file:
    df_topics.to_csv(file)


# Compute Model Perplexity and Coherence Score: This model to judge how good the model performed,
# especially by Coherene score
# Compute Coherence Score
coherence_ldamodel = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_ldamodel.get_coherence()
print('\nCoherence Score: ', coherence_lda)