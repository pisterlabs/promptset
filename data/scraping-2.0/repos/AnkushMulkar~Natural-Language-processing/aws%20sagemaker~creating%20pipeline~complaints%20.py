#  pre-processing steps to assign the topics and create TFIDF feature vectors for model training 

f = open('complaints.json') 
import json
import pandas as pd
  
# returns JSON object as  
# a dictionary 
data = json.load(f)
df=pd.json_normalize(data)


# Import required libs

!pip install spacy
!python -m spacy download en_core_web_sm

import json 
import numpy as np
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
import nltk 

nlp = en_core_web_sm.load()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint

df.to_csv('complaints.csv', encoding='utf-8', index=False)

# Assign coloumn names 

df.columns = ["index","type","id","score","tags","zip_code","complaint_id","issue","date_received","state","consumer_disputed","product","company_response","company","submitted_via","date_sent_to_company","company_public_response","sub_product","timely","complaint_what_happened","sub_issue","consumer_consent_provided"]

#Assigning nan in place of blanks in the complaints column(complaint_what_happened)
df[df['complaint_what_happened']==''] = np.nan

# Fucntion to clean the text 
def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df_clean = pd.DataFrame(df.complaint_what_happened.apply(lambda x: clean_text(x)))

import re, nltk, spacy, string
pd.options.mode.chained_assignment = None  
df.complaint_what_happened=df.complaint_what_happened.astype(str)


df_clean = pd.DataFrame(df.complaint_what_happened.apply(lambda x: clean_text(x)))

#Function to Lemmatize the texts
def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

#Creating a dataframe that will have only the complaints and the lemmatized complaints.
import en_core_web_sm
nlp = en_core_web_sm.load()
df_clean["Complaint_lemmatize"] =  df_clean.apply(lambda x: lemmatizer(x['complaint_what_happened']), axis=1)

#Using custom Chunking
#Chunking in NLP is a process to take small pieces of information and group them into large units. The primary use of Chunking is making groups of "noun phrases.
#Here we are using only noun, singular as we have already lemmatized the texts.
import pandas as pd
!pip install TextBlob
from textblob import TextBlob
nltk.download('punkt')

def pos_tag(text):
    try:
        return TextBlob(text).tags
    except:
        return None

def get_adjectives(text):
    blob = TextBlob(text)
    return ' '.join([ word for (word,tag) in blob.tags if tag == "NN"])

df_clean["complaint_POS_removed"] =  df_clean.apply(lambda x: get_adjectives(x['Complaint_lemmatize']), axis=1)

#Removing -PRON- from the text corpus
df_clean['Complaint_clean'] = df_clean['complaint_POS_removed'].str.replace('-PRON-', '')

df_clean['Complaint_clean'] = df_clean['Complaint_clean'].str.replace('xxxx','')

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = tfidf.fit_transform(df_clean['Complaint_clean'])

import warnings
!pip install gensim==4.1.0
warnings.filterwarnings("ignore")
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from gensim.models.coherencemodel import CoherenceModel
from operator import itemgetter
# Use Gensim's NMF to get the best num of topics via coherence score
texts = df_clean['Complaint_clean']
dataset = [d.split() for d in texts]

# Create a dictionary
# In gensim a dictionary is a mapping between words and their integer id
dictionary = Dictionary(dataset)

# Filter out extremes to limit the number of features
dictionary.filter_extremes(
    no_below=3,
    no_above=0.85,
    keep_n=5000
)

# Create the bag-of-words format (list of (token_id, token_count))
corpus = [dictionary.doc2bow(text) for text in dataset]

# Create a list of the topic numbers we want to try
topic_nums = list(np.arange(5, 10, 1))

# Run the nmf model and calculate the coherence score
# for each number of topics
coherence_scores = []

for num in topic_nums:
    nmf = Nmf(
        corpus=corpus,
        num_topics=num,
        id2word=dictionary,
        chunksize=2000,
        passes=5,
        kappa=.1,
        minimum_probability=0.01,
        w_max_iter=300,
        w_stop_condition=0.0001,
        h_max_iter=100,
        h_stop_condition=0.001,
        eval_every=10,
        normalize=True,
        random_state=42
    )
    
    # Run the coherence model to get the score
    cm = CoherenceModel(
        model=nmf,
        texts=texts,
        dictionary=dictionary,
        #coherence='c_v'
    )
    
    coherence_scores.append(round(cm.get_coherence(), 5))

# Get the number of topics with the highest coherence score
scores = list(zip(topic_nums, coherence_scores))
best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]

print(best_num_topics)

from sklearn.decomposition import NMF
nmf_model = NMF(n_components=5,random_state=40)

nmf_model.fit(dtm)
len(tfidf.get_feature_names())

#Print the top word of a sample component
single_topic = nmf_model.components_[0]
single_topic.argsort()
top_word_indices = single_topic.argsort()[-10:]
for index in top_word_indices:
    print(tfidf.get_feature_names()[index])
    
    #Print Top15 words for each of the topics
for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    
    #Creating the best topic for each complaint
topic_results = nmf_model.transform(dtm)
topic_results[0].round(2)
topic_results[0].argmax()
topic_results.argmax(axis=1)

#Assign the best topic to each of the cmplaints in Topic Column
df_clean['Topic'] = topic_results.argmax(axis=1)

#Create the dictionary of Topic names and Topics
Topic_names = {0:"Bank Account services",1:"Credit card or prepaid card", 2:"Others",3:"Theft/Dispute Reporting",4:"Mortgage/Loan"}

#Replace Topics with Topic Names
df_clean['Topic'] = df_clean['Topic'].map(Topic_names)

df_clean.to_csv('complaintsclean.csv', encoding='utf-8', index=False)

Topic_names = {"Bank Account services":0,"Credit card or prepaid card":1,"Others":2,"Theft/Dispute Reporting":3,"Mortgage/Loan":4}
df_clean['Topic'] = df_clean['Topic'].map(Topic_names)

training_data=df_clean[["complaint_what_happened","Topic"]]

df_clean.to_csv('complaintsTraining.csv', encoding='utf-8', index=False)

import pandas as pd

data = pd.read_csv('complaintsTraining.csv')

training_data=data[["complaint_what_happened","Topic"]]

training_data.to_csv('complaintssharpened.csv', encoding='utf-8', index=False)

import pickle
from sklearn.feature_extraction.text import CountVectorizer


#GET VECTOR COUNT
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data['complaint_what_happened'].values.astype('U'))

pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))

from sklearn.feature_extraction.text import TfidfTransformer

#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

df.shape
df.to_csv("tfidf_features.csv",encoding='utf-8', index=False)

df= df.assign(lables=training_data[:1000]['Topic'])

ylabel=training_data[:1000]['Topic']

df= df.assign(lables=ylabel)

df['lables']=ylabel.values

df.to_csv('tfidffeatureswithlables.csv', encoding='utf-8', index=False)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, df.lables, test_size=0.25, random_state=42)

# Remove lable coloumn from test data 
del X_test[X_test.columns[-1]]

# Write the split data into csv for further processing 
y_train.to_csv('y_train_features.csv', encoding='utf-8', index=False)
y_test.to_csv('y_test_features.csv', encoding='utf-8', index=False)
X_train.to_csv('X_train_features.csv', encoding='utf-8', index=False)
X_test.to_csv('X_test_features.csv', encoding='utf-8', index=False)