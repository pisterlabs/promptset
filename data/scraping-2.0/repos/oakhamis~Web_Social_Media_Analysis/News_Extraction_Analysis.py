"""
This script performs news extraction and analysis.
"""

# Importing necessary libraries and packages
import pandas as pd
from pandas import json_normalize
pd.options.display.max_columns = 200
pd.options.mode.chained_assignment = None

import requests
from newsapi import NewsApiClient
import re
from io import StringIO
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer    
stop = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import Counter
from functools import reduce
import datetime
from datetime import datetime, timedelta
import warnings
pd.set_option('display.max_colwidth', -1)
import bs4 as bs 
import urllib.request 
from urllib.error import URLError

from string import punctuation
from functools import reduce 
from matplotlib import pyplot as plt
from collections import Counter
import re

from tqdm import tqdm_notebook
tqdm_notebook().pandas()

# Suppress warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings('ignore')

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import spacy
from textblob import TextBlob 
import texttable as tt
from wordcloud import WordCloud

import os
import sys
if sys.version_info[0] >= 3:
    unicode = str

os. getcwd()

# Set working directory
os.chdir('C:/Users\Omar\Desktop\Spring 2021\Web & Social Media Analytics\Assessment\Final_code')

# Initializing the News API client
api_key='4b91ff37253c4127bde291a1f763642a'
newsapi = NewsApiClient(api_key=api_key)

# Function to retrieve the sources of news
def getSources():
    source_url = 'https://newsapi.org/v1/sources?language=en'
    response = requests.get(source_url).json()
    sources = []
    for source in response['sources']:
        sources.append(source['id'])
    return sources

all_sources = getSources()

# Function to generate dates for a year
import datetime
from datetime import datetime, timedelta
def date(base):    
    date_list=[]    
    yr=datetime.today().year    
    if (yr%400)==0 or ((yr%100!=0) and (yr%4==0)):          
        numdays=366        
        date_list.append([base - timedelta(days=x) for x in    
        range(366)])   
    else:        
        numdays=365        
        date_list.append([base - timedelta(days=x) for x in    
        range(365)])    
        newlist=[]    
        for i in date_list:        
           for j in sorted(i):            
               newlist.append(j)    
        return newlist 

# Function to generate the last 30 dates
def last_30(base):     
    date_list=[base - timedelta(days=x) for x in range(30)]      
    return sorted(date_list)

# Function to generate 'from' date.
def from_dt(x):    
    from_dt=[]    
    for i in range(len(x)):          
        from_dt.append(last_30(datetime.today())[i-1].date())         
    return from_dt

# Function to generate 'to' date
def to_dt(x):    
    to_dt=[]    
    for i in range(len(x)):        
        to_dt.append(last_30(datetime.today())[i].date())    
    return to_dt


from_list=from_dt(last_30(datetime.today()))
to_list=to_dt(last_30(datetime.today()))
 
# Function to extract news articles based on query within a date range    
def func(query):
    newdf=pd.DataFrame()
    
    for (from_dt,to_dt) in zip(from_list,to_list):
        all_articles = newsapi.get_everything(q=query,language='en', sources=','.join(all_sources), sort_by='relevancy', from_param=from_dt,to=to_dt)
        d=json_normalize(all_articles['articles'])
        newdf=newdf.append(d)
    
    return newdf    

responses = {
    100: ('Continue', 'Request received, please continue'),
    101: ('Switching Protocols',
          'Switching to new protocol; obey Upgrade header'),

    200: ('OK', 'Request fulfilled, document follows'),
    201: ('Created', 'Document created, URL follows'),
    202: ('Accepted',
          'Request accepted, processing continues off-line'),
    203: ('Non-Authoritative Information', 'Request fulfilled from cache'),
    204: ('No Content', 'Request fulfilled, nothing follows'),
    205: ('Reset Content', 'Clear input form for further input.'),
    206: ('Partial Content', 'Partial content follows.'),

    300: ('Multiple Choices',
          'Object has several resources -- see URI list'),
    301: ('Moved Permanently', 'Object moved permanently -- see URI list'),
    302: ('Found', 'Object moved temporarily -- see URI list'),
    303: ('See Other', 'Object moved -- see Method and URL list'),
    304: ('Not Modified',
          'Document has not changed since given time'),
    305: ('Use Proxy',
          'You must use proxy specified in Location to access this '
          'resource.'),
    307: ('Temporary Redirect',
          'Object moved temporarily -- see URI list'),

    400: ('Bad Request',
          'Bad request syntax or unsupported method'),
    401: ('Unauthorized',
          'No permission -- see authorization schemes'),
    402: ('Payment Required',
          'No payment -- see charging schemes'),
    403: ('Forbidden',
          'Request forbidden -- authorization will not help'),
    404: ('Not Found', 'Nothing matches the given URI'),
    405: ('Method Not Allowed',
          'Specified method is invalid for this server.'),
    406: ('Not Acceptable', 'URI not available in preferred format.'),
    407: ('Proxy Authentication Required', 'You must authenticate with '
          'this proxy before proceeding.'),
    408: ('Request Timeout', 'Request timed out; try again later.'),
    409: ('Conflict', 'Request conflict.'),
    410: ('Gone',
          'URI no longer exists and has been permanently removed.'),
    411: ('Length Required', 'Client must specify Content-Length.'),
    412: ('Precondition Failed', 'Precondition in headers is false.'),
    413: ('Request Entity Too Large', 'Entity is too large.'),
    414: ('Request-URI Too Long', 'URI is too long.'),
    415: ('Unsupported Media Type', 'Entity body in unsupported format.'),
    416: ('Requested Range Not Satisfiable',
          'Cannot satisfy request range.'),
    417: ('Expectation Failed',
          'Expect condition could not be satisfied.'),

    500: ('Internal Server Error', 'Server got itself in trouble'),
    501: ('Not Implemented',
          'Server does not support this operation'),
    502: ('Bad Gateway', 'Invalid responses from another server/proxy.'),
    503: ('Service Unavailable',
          'The server cannot process the request due to a high load'),
    504: ('Gateway Timeout',
          'The gateway server did not receive a timely response'),
    505: ('HTTP Version Not Supported', 'Cannot fulfill request.'),
    }

# Function to scrape the full text of news articles from the URLs
def get_url_text(NewsDF):
     
    Fulltext = []
    for index, row in NewsDF.iterrows():  
        
        try:
            scraped_data = urllib.request.urlopen(row['url'])
        except urllib.error.HTTPError as e:
            print(e.code)
            print(e.read())
        except URLError as u:
            print('We failed to reach a server.')
            print('Reason: ', u.reason)
        
        article = scraped_data.read()
    
        parsed_article = bs.BeautifulSoup(article,'lxml')
        
        paragraphs = parsed_article.findAll("p")
        
        article_text = ""
        for p in paragraphs:
            article_text += p.text
    
        
        Fulltext.append(article_text)
        
    NewsDF["Fulltext"] = Fulltext    
        
    return NewsDF

# Data extraction and cleaning
df = pd.DataFrame(func('aapi'))
df = df.drop_duplicates('description')
df = df.drop_duplicates('urlToImage')
df_final = get_url_text(df)
df_final['Fulltext'].replace('',np.nan,inplace=True)
df_final.dropna(subset=['Fulltext'],inplace=True)
df_final = df_final.drop_duplicates('Fulltext')
df_final = df_final[df_final['source.name'] != "CNN"]
df_final.to_csv('aapi_news.csv', index=False)
df_final.head()

# Data loading and basic analysis
data = pd.read_csv('aapi_news_3103_2904.csv')
data = data.drop(columns='urlToImage')
data.reset_index(inplace=True, drop=True)
print(data.shape)
title = list(data['title'])

words=[]

for t in title:
    res = len(t.split())
    words.append(res)

word_title=pd.DataFrame({"HISTOGRAM FOR WORD COUNT BY TITLE":words},index=words)
word_title.hist(figsize=(15, 5), bins=12)

type(data['publishedAt'])

data['source.name'].value_counts(normalize=False).plot(kind='bar', grid=True, 
                                                       figsize=(10, 5), color = 'b')
stop_words = []
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','said','would','wa','ha','u', 'say','also','mr','mrs,ms'])

# Function to remove stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stop_words]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)   
    return filtered_text

# Function to remove non-Ascii characters
def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    
    text = _removeNonAscii(text)
    text = remove_stopwords(text)
    text = text.strip()
    return text

# Text cleaning, tokenization and summarization
nlp = spacy.load("en_core_web_sm")
data["Sentences"] = data["Fulltext"].apply(lambda x: [sent.text for sent in nlp(x).sents])

# List of sentences to be removed as they are not relevent to the main text
sentence_list = []
for sublist in data['Sentences']:
    for item in sublist:
        sentence_list.append(item)
        
#Remove sentences with freq. more than 2        
sentence_list= [ i for i in sentence_list if sentence_list.count(i) > 2]

data['Cleaned_Sentences'] = (data['Sentences'].explode().loc[lambda x: ~x.isin(sentence_list)].groupby(level=0).agg(list))

Clean_sentence_list = []
for sublist in data['Cleaned_Sentences']:
    for item in sublist:
        Clean_sentence_list.append(item)

sentence_frequencies = {}
for sent in Clean_sentence_list:
    if sent not in sentence_frequencies.keys():
        sentence_frequencies[sent] = 1
    else:
        sentence_frequencies[sent] += 1 

# Combine all sentences from all news in one summarized text
combined_Sentences=' '.join(Clean_sentence_list)

combined_text=clean_text(combined_Sentences) 

word_frequencies = {}
for word in nltk.word_tokenize(combined_text):
    if word not in stop_words:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

sentence_scores = {}
for sent in Clean_sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

import heapq
summary_sentences = heapq.nlargest(10, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)
print(summary)

# Another method of summarization
from gensim.summarization.summarizer import summarize 
gensim_summary = summarize(combined_Sentences, word_count= 200)


def clean_Fulltext(text):
    corpus=[]
    for sentence in text:
        sent=sentence
        sent=' '.join(sent)
        corpus.append(sent)
    return corpus

data['Clean_Fulltext']=clean_Fulltext(data.Cleaned_Sentences.values) 

# Subjectivity and Sentiment Analysis
subjectivity_list = []
polarity_list = []

for index, row in data.iterrows():    
    analysis = TextBlob(row['Fulltext'])
    subjectivity = analysis.sentiment.subjectivity
    subjectivity_list.append(subjectivity)
    polarity = analysis.sentiment.polarity
    polarity_list.append(polarity)
    
tab = tt.Texttable()
headings = ['source.name','Subjectivity','Polarity']
tab.header(headings)

for row in zip(data['source.name'], subjectivity_list, polarity_list):
    tab.add_row(row)

avg_subjectivity = (sum(subjectivity_list) / len(subjectivity_list))
avg_polarity = (sum(polarity_list) / len(polarity_list))
table = tab.draw()
text = table
df = pd.read_csv(StringIO(re.sub(r'[-+|]', '', text)), sep='\s{2,}', engine='python')
df = df[df['Subjectivity'].notna()]
df = df.groupby("source.name").mean()
print(df)
print (table)
print (data['source.name'] +  'average subjectivity: ' + str(avg_subjectivity))
print (data['source.name'] + ' average polarity: ' + str(avg_polarity))


# Lemmatization and Keyword Extraction
def lemmatize_text(column):
    corpus=[]
    lemma = WordNetLemmatizer()
    for i in range(0,len(column)):
        text=str(column[i])
        text=re.sub('[^a-zA-Z]',' ',text)
        text=[lemma.lemmatize(w) for w in word_tokenize(str(text).lower())]
        text=' '.join(text)
        corpus.append(text)
    return corpus

 
data['Clean_Fulltext']=lemmatize_text(data.Clean_Fulltext.values)

#Function to tokenize words in sentences
def word_tokenizer(text):
    text = clean_text(text)
    tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]
    tokens = list(reduce(lambda x,y: x+y, tokens))
    tokens = list(filter(lambda token: token not in (stop_words + list(punctuation)) , tokens))
    return tokens

data['tokens'] = data['Clean_Fulltext'].progress_map(lambda d: word_tokenizer(d)) 

# Function to gather keywords
def keywords(source):
    tokens = data[data['source.name'] == source]['tokens']
    alltokens = []
    for token_list in tokens:
        alltokens += token_list
    counter = Counter(alltokens)
    return counter.most_common(10)

word_list = []
for sublist in data['tokens']:
    for item in sublist:
        word_list.append(item)

frequency_dist = nltk.FreqDist(word_list)
frequency_dist.plot(30,cumulative=False)

for source in set(data['source.name']):
    print('source.name :', source)
    print('top 10 keywords:', keywords(source))
    print('---')

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(min_df=5, analyzer='word', ngram_range=(1, 3), stop_words='english')
vz = vectorizer.fit_transform(list(data['tokens'].map(lambda tokens: ' '.join(tokens))))
vz.shape

tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
tfidf.columns = ['tfidf']
tfidf.tfidf.hist(bins=25, figsize=(15,7))

# Defining the word cloud and vizualizing it
def plot_word_cloud(terms):
    text = terms.index
    text = ' '.join(list(text))
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure(figsize=(25, 25))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
   
plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=True).head(40))

# Topic Modelling
aux = data.copy()
id2word = corpora.Dictionary(aux['tokens'])
texts = aux['tokens'].values
corpus = [id2word.doc2bow(text) for text in texts]

def LDA_model(num_topics, passes=1):
    return gensim.models.ldamodel.LdaModel(corpus=tqdm_notebook(corpus, leave=False),
                                               id2word=id2word,
                                               num_topics=num_topics,
                                               random_state=100,
                                               eval_every=10,
                                               chunksize=2000,
                                               passes=passes,
                                               per_word_topics=True
                                            )


def compute_coherence(model):
    coherence = CoherenceModel(model=model,
                           texts=aux['tokens'].values,
                           dictionary=id2word, coherence='c_v')
    return coherence.get_coherence()

def display_topics(model):
    topics = model.show_topics(num_topics=model.num_topics, formatted=False, num_words=10)
    topics = map(lambda c: map(lambda cc: cc[0], c[1]), topics)
    df = pd.DataFrame(topics)
    df.index = ['topic_{0}'.format(i) for i in range(model.num_topics)]
    df.columns = ['keyword_{0}'.format(i) for i in range(1, 10+1)]
    return df

def explore_models(df, rg=range(5, 25)):
    id2word = corpora.Dictionary(df['tokens'])
    texts = df['tokens'].values
    corpus = [id2word.doc2bow(text) for text in texts]
    models = []
    coherences = []
   
    for num_topics in tqdm_notebook(rg, leave=False):
        lda_model = LDA_model(num_topics, passes=5)
        models.append(lda_model)
        coherence = compute_coherence(lda_model)
        coherences.append(coherence)
     
    fig = plt.figure(figsize=(15, 5))
    plt.title('Choosing the optimal number of topics')
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence')
    plt.grid(True)
    plt.plot(rg, coherences)
   
    return coherences, models

coherences, models = explore_models(aux, rg=range(5, 25, 5))
best_model = LDA_model(num_topics=10, passes=5)
feature_names = vectorizer.get_feature_names()
no_top_words = 10
topic_keywords = display_topics(model=best_model)

def get_document_topic_matrix(corpus, num_topics=best_model.num_topics):
    matrix = []
    for row in tqdm_notebook(corpus):
        output = np.zeros(num_topics)
        doc_proba = best_model[row][0]
        for doc, proba in doc_proba:
            output[doc] = proba
        matrix.append(output)
    matrix = np.array(matrix)
    return matrix

matrix = get_document_topic_matrix(corpus)

# Visualizating the topic model
pyLDAvis.enable_notebook()
panel = gensimvis.prepare(best_model, corpus, id2word)
pyLDAvis.save_html(panel, 'pyLDAvis.html')