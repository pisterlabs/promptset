#%%
# basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import jsonlines
from dotenv import load_dotenv
import os 
# bertopic
from bertopic import BERTopic
from bertopic.representation import OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.backend import BaseEmbedder

from sentence_transformers import SentenceTransformer
import openai
from hdbscan import HDBSCAN
from umap import UMAP

from gsdmm import MovieGroupProcess

# preprocessing 
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")    

#%%
# given the path of a directory containing jsonl files, (tweets extracted with twarc2)
# return a dataframe with the tweets
# every jsonl file is a topic

def get_test_dataset(path: str):
    print('getting dataset')
    df = pd.DataFrame(columns=['text', 'lang', 'topic'])

    for file in os.listdir(path): # read files in directory
        if file.endswith(".jsonl"):
            with jsonlines.open(os.path.join(path,file)) as reader: # open file
                data = list(reader)
                for batch in data:  # every line contain 100 tweets
                    for tweet in batch['data']:
                        df.loc[tweet['id']] = [tweet['text'], tweet['lang'], file[:-6]]



    df['text'] = df['text'].str.replace(r'RT', '', regex=True, case=False) # remove RT
    df['text'] = df['text'].str.replace(r'\n', '', regex=True, case=False) # remove \n
    df['text'] = df['text'].str.replace(r'http\S+', '', regex=True, case=False) # remove urls

    df = df[df['lang'] == 'en'] # remove non english tweets

    df_nohash = df.copy()
    df_nohash['text'] = df_nohash['text'].str.replace(r'#\S+', '', regex=True, case=False)

    return df, df_nohash

# this is required for traditional methods 
def preprocess_text(text):

    text = re.sub(r'http\S+', '', text) # remove urls
    text = text.lower()                 # lowercase
    text = re.sub(r'@\S+', '', text)        # remove mentions
    text = re.sub(r'#', '', text)       # remove hashtags
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = re.sub("(\\d|\\W)+"," ",text)        # remove numbers
    text = text.strip()                 # remove whitespaces
    text = contractions.fix(text)           # expand contractions
    text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))]) # remove stopwords
    text = text.replace('amp', '')   # remove amp
    text = ' '.join([word for word in text.split() if len(word) > 2])#remove 2 letter words

    return text



class Evaluator:

    def __init__(self, dataset, name, nr_topics = 'auto', min_topic_size = 50, n_iter=2):
        '''
        dataset: dataframe with the tweets
        name: name of the embedding model, if name is openai use openai api ( openai, NMF, GSDMM, every sentence transformer model)
        nr_topics: number of topics to infer
        min_topic_size: minimum size of a topic
        n_iter: number of iterations to run the model
        '''

        self.df = dataset.copy()
        self.docs = self.df['text'].tolist()
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        self.n_iter = n_iter

        self.embeddings = None
        self.embedder = None
        self.model = None
        self.accuracy = []                 # accracy result
        self.accuracy_no_outliers = []     # accuracy without outliers 
        self.name = name
        self.topic_share = []

        self.evaluate()             # create embeddings 
        self.get_accuracy()         # get accuracy

    # given name of the model compute the embeddings, if name is openai use openai api 
    # else only sentence tranformer are allowed
    def evaluate(self):
        print('evaluate', self.name)
        model = 'bertopic' if self.name != 'NMF' and self.name != 'GSDMM' and self.name != 'openai' else self.name

        print('model ', model)

        # create embeddings, if we use openai we have to use the api, in particular the text-embedding-ada-002 model
        if(self.name == 'openai'):
            print('embeddings with openai')

            embs = openai.embeddings.create(input = self.docs, model="text-embedding-ada-002").data
            self.embeddings = np.array([np.array(emb.embedding) for emb in embs])
            model = 'bertopic'           # create model
           
        elif(model == 'bertopic'):
            print('embeddings with bertopic')
            self.embedder = SentenceTransformer(self.name)
            self.embeddings = self.embedder.encode(self.docs)
            model = 'bertopic'           # create model


        
        model_methods = {
            'bertopic': self.get_BERTopic_model,
            'NMF': self.get_NMF,
            'GSDMM': self.get_GSDMM
        }

        # fit the model and get accuracy for n times 
        for i in range(self.n_iter):
            print('iteration ', i)
            if model in model_methods:
                model_methods[model](n=i)  # create model ( takes the model from the dictionary model_methods)
                #self.get_accuracy()  # get accuracy
            

    # create topic model with bertopic and update the dataframe with the inferred topics 
    def get_BERTopic_model(self, n):
        print('fitting bertopic model')

        vectorizer_model = CountVectorizer(stop_words="english")
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        model = BERTopic( 
                            vectorizer_model =   vectorizer_model,
                            ctfidf_model      =   ctfidf_model,
                            nr_topics        =   self.nr_topics,
                            min_topic_size   =   self.min_topic_size,
                            embedding_model  =   self.embedder
                        )
        topics, probs = model.fit_transform(self.docs, embeddings = self.embeddings)

        self.model = model
        self.df['my_topics_'+str(n)] = topics
        #self.df['my_probs'] = probs

        return model

    def get_NMF(self, n,  max_df = 0.95, min_df = 3, ngram_range = (1,2)  ):
        
        print('fitting NMF model, max_df = ', max_df, ' min_df = ', min_df, ' ngram_range = ', ngram_range)

        self.df['preprocessed'] = self.df['text'].apply(preprocess_text) # we need to preprocess the text for NMF
        tfidf = TfidfVectorizer(stop_words='english', max_df = max_df, min_df = min_df, ngram_range = ngram_range)
        dtm = tfidf.fit_transform(self.df['preprocessed'])


        nmf_model = NMF(n_components= len(self.df['topic'].unique()), random_state=42)

        topics = nmf_model.fit_transform(dtm)



        self.model = nmf_model
        self.df['my_topics_'+str(n)] = topics.argmax(axis=1)

    def get_GSDMM(self,n = 0, alpha = 0.1, beta = 0.1, n_iters = 30):

        if self.nr_topics == 'auto':
            self.nr_topics = len(self.df['topic'].unique())

        mgp = MovieGroupProcess(K= self.nr_topics, alpha= alpha, beta=beta, n_iters=n_iters)
        lemmatizer = nltk.WordNetLemmatizer()
        vectorizer = TfidfVectorizer()

        self.df['tokens'] = self.df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
        self.df['tokens'] = self.df['tokens'].apply(lambda x: nltk.word_tokenize(x))
        self.df['tokens'] = self.df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        docs = self.df['tokens'].tolist()
        vocab = set(x for doc in docs for x in doc)
        n_terms = len(vocab)
        y = mgp.fit(docs, n_terms)

        self.model = mgp
        self.df['my_topics_'+str(n)] = y



    def get_accuracy(self):
        df = self.df


        for i in range(self.n_iter):

            topics = df['topic'].unique()
            my_topics = df['my_topics_'+str(i)].unique()
            results = {}
            results_no_outliers = {}
            topic_share = {}
            # every my_topic should have >77 % pf the documents of topic
            print(df)
            for topic in my_topics:
                res = df[df['my_topics_'+str(i)] == topic].value_counts('topic') 
                if topic != -1:
                    topic_share[topic] = round(res[0]/ sum(res) ,2)
                    

            # compute accuracy for each topic
            if (min(topic_share.values()) < 0.7):
                print('topic share is too low, proprablt accuracy is not meaningful for ', min(topic_share))
                print('check the heatmap ')
            results['min_topic_share'] = min(topic_share.values())

            for topic in topics:
                res = df[df['topic'] == topic].value_counts('my_topics_'+str(i))
                first = res.iloc[0] if res.index[0] != -1 else res.iloc[1]                       # i'm assuming that out of the possible label the right one is the biggest 
                missed = sum(res.iloc[1:]) if res.index[0] != -1 else sum(res) - res.iloc[1]    # sum of the other labels
                try :
                    outliers = res.loc[-1]
                except:
                    outliers = 0

                
                
                results[topic] = first / (first + missed)
                results_no_outliers[topic] = first / (first + missed - outliers)  # bertopic mark the outliers with -1, i do not consider them while computing accuracy
            
            print('accuracy ', results)
            self.accuracy.append({
                'accuracy': results,
                'accuracy_no_outliers': results_no_outliers,
                'topic_share': topic_share
            }) 
            

        return results
    
    def visualize_documents(self):
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(self.embeddings)
        return self.model.visualize_documents(self.docs, reduced_embeddings=reduced_embeddings)

    def visualize_heatmap(self, n=0, ax=None):
        if ax is None:
            ax = plt.gca()
        sns.heatmap(pd.crosstab(self.df['topic'], self.df['my_topics_'+str(n)]), annot=True, cmap="YlGnBu", fmt='g', ax=ax)
        ax.set_title(self.name)
        return ax
        
    def visualize_min_topic_share(self):

        # create a figure and axis
        min_topic_size = [acc['accuracy']['min_topic_share'] for acc in self.accuracy]

        fig, ax = plt.subplots()
        #bar plot
        ax.bar(range(len(min_topic_size)), min_topic_size) 

        ax.set_xlabel('iteration')
        ax.set_ylabel('min topic share')
        ax.set_title( self.name)

        # y lim 0-1 
        ax.set_ylim(0,1)

        # x ticks
        ax.set_xticks(range(len(min_topic_size)))

    #static method 
    @staticmethod
    def compare_models(models):
    # barplot grouped by column

        pd.DataFrame(models).T.plot.bar(rot=0, figsize=(10,5))
        #add title
        plt.title('Accuracy of the models')
        #add grid every 0.1
        plt.grid(axis='y', alpha=0.5)
        #add yticks every 0.1
        plt.yticks(np.arange(0, 1.1, 0.1))
        # legend outside 
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    



# %%
# find differences between columns 
