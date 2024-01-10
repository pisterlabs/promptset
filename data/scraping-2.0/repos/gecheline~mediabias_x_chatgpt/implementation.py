import streamlit as st
import pandas as pd
import numpy as np
from umap import UMAP
import plotly.express as px
import os
import pickle
import datetime

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import openai
import tiktoken
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY') 
openai.api_key = api_key


@st.cache_resource
class MediaBiasModel:

    def __init__(self, label='mediabiasmodel'):
        '''Initializes a model and loads the media bias dataset.'''
        self.label='mediabiasmodel'

        # set the embedding model parameters
        self.embedding_model = "text-embedding-ada-002"
        embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
        self.max_tokens = 8000
        self.encoding = tiktoken.get_encoding(embedding_encoding)


    def load_data(self):
        '''
        Loads the dataset required for training the model and remaps the topics and media bias to simplified labels.
        '''

        media_bias_map = {
            'HuffPost': 'left',
            'Federalist': 'right',
            'Daily Beast': 'left',
            'Alternet': 'left',
            'Breitbart': 'right',
            'New Yorker': 'left',
            'American Greatness': 'right', # from https://mediabiasfactcheck.com/american-greatness/
            'Daily Caller': 'right',
            'Daily Wire': 'right',
            'Slate': 'left',
            'Reuters': 'center',
            'Hill': 'center', # from https://mediabiasfactcheck.com/the-hill/
            'USA Today': 'left',
            'CNBC': 'left',
            'Yahoo News - Latest News & Headlines': 'left',
            'AP': 'left',
            'Bloomberg': 'left',
            'Fox News': 'right',
            'MSNBC': 'left',
            'Daily Stormer': 'right', # from https://mediabiasfactcheck.com/the-hill/
            'New York Times': 'left'
            }

        new_topics_map = {
        '#metoo':'activism',
        'abortion':'abortion',
        'black lives matter':'activism',
        'blm':'activism',
        'coronavirus':'coronavirus-and-vaccines',
        'elections-2020':'politics',
        'environment':'environment',
        'gender':'socioeconomics',
        'gun control':'gun-control',
        'gun-control':'gun-control',
        'immigration':'immigration',
        'international-politics-and-world-news':'politics',
        'islam':'islam',
        'marriage-equality':'activism',
        'middle-class':'socioeconomics',
        'sport':'sport',
        'student-debt':'socioeconomics',
        'taxes':'socioeconomics',
        'trump-presidency':'politics',
        'universal health care':'universal-health-care',
        'vaccine':'coronavirus-and-vaccines',
        'vaccines':'coronavirus-and-vaccines',
        'white-nationalism':'white-nationalism',
        }

        df = pd.read_excel('data/final_labels_SG2.xlsx')
        df = df[df['label_bias']!='No agreement']
        df['outlet_bias'] = df['outlet'].map(media_bias_map)
        df = df.drop(columns=['type'])

        df = df.rename(columns={'topic':'topic_original'})
        df['topic'] = df['topic_original'].map(new_topics_map)
        
        self.df = df

    def transform_data(self, input_df=None):
        '''Transforms the sentence data with OpenAI ADA embeddings.'''
        # read the API KEY from an environment variable and set it in openai
        if input_df is None:
            if 'embedding' in self.df.columns:
                return self.df
            else:
                # omit reviews that are too long to embed
                self.df["n_tokens"] = self.df.text.apply(lambda x: len(self.encoding.encode(x)))
                self.df = self.df[self.df.n_tokens <= self.max_tokens]

                # get embedding
                self.df["embedding"] = self.df.text.apply(lambda x: get_embedding(x, engine=self.embedding_model))
                return self.df
        else:
            input_df["n_tokens"] = input_df.text.apply(lambda x: len(self.encoding.encode(x)))
            input_df = input_df[input_df.n_tokens <= self.max_tokens]

            # get embedding
            input_df["embedding"] = input_df.text.apply(lambda x: get_embedding(x, engine=self.embedding_model))
            return input_df

    
    def get_text_embedding(self, text):
        '''Gets the embedding of a single sentence.'''

        n_tokens = len(self.encoding.encode(text))
        print(n_tokens)
        if n_tokens > self.max_tokens:
            raise ValueError(f'The input text is too long and resulting in {n_tokens} tokens. The maximum number of tokens for this model is 8000.')
        
        return get_embedding(text, engine=self.embedding_model)

    def fit_models(self):
        '''
        Fits the classifiers for topic, bias and outlet bias on the training data.
        '''
        self.topic_model = KNeighborsClassifier(metric='euclidean', n_neighbors=10, weights='distance')
        self.bias_model = LogisticRegression(C=2.782559402207126, max_iter=1000, solver='newton-cg', penalty='l2')
        self.politics_model = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=300)

        print('Fitting topic classifier...')
        self.topic_model.fit(np.vstack(self.df.embedding.values), self.df.topic)
        print('Fitting bias classifier...')
        self.bias_model.fit(np.vstack(self.df.embedding.values), self.df.label_bias)
        print('Fitting political bias classifier...')
        self.politics_model.fit(np.vstack(self.df.embedding.values), self.df.outlet_bias)
        
        print('Model fitting complete. To save the current classifiers, run model.save().')

    def load_models(self, model_tag, directory='models'):
        '''
        Loads pre-trained topic, bias and political bias models. 
        Requires a user-specified model tag to correctly identify models.
        '''
        self.model_tag = model_tag
        self.topic_model = pickle.load(open(f'{directory}/model_topic_{model_tag}.pickle', 'rb'))
        self.bias_model = pickle.load(open(f'{directory}/model_bias_{model_tag}.pickle', 'rb'))
        self.politics_model = pickle.load(open(f'{directory}/model_politics_{model_tag}.pickle', 'rb'))
        

    def save_models(self, model_tag=None, directory='models'):
        '''
        Saves the model to a pickle file and makes it available for reuse without re-training.
        '''
        if model_tag is None:
            # get a datetime tag to assign to model
            model_tag = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")

        self.model_tag = model_tag
        pickle.dump(self.topic_model, open(f'{directory}/model_topic_{model_tag}.pickle', 'wb'))
        pickle.dump(self.bias_model, open(f'{directory}model_bias_{model_tag}.pickle', 'wb'))
        pickle.dump(self.politics_model, open(f'{directory}/model_politics_{model_tag}.pickle', 'wb'))

    def predict_labels_text(self, sentence):
        '''
        Predicts the topic, bias and outlet bias of a single sentence.
        '''
        embedding = np.array(self.get_text_embedding(sentence))
        topic = self.topic_model.predict(embedding.reshape(1, -1))
        bias = self.bias_model.predict(embedding.reshape(1, -1))
        politics = self.politics_model.predict(embedding.reshape(1, -1))

        return topic, bias, politics

    def predict_labels_df(self, input_df):
        '''
        Transforms text into embeddings and predicts the topic, bias and outlet bias of the 'text' column of an input dataframe.
        '''
        input_df = self.transform_data(input_df=input_df)
        input_df['topic'] = self.topic_model.predict(np.vstack(input_df.embedding.values))
        input_df['label_bias'] = self.bias_model.predict(np.vstack(input_df.embedding.values))
        input_df['outlet_bias'] = self.politics_model.predict(np.vstack(input_df.embedding.values))

        return input_df

    def analyze_full_article(self, article_text):
        '''
        Splits the input article text into sentences and analyzes each one separately for bias.
        '''
        # transform the article into sentence data and get labels
        sentences = article_text.replace('\n','').replace('U.S.','United States').split('.')
        df_sentences = pd.DataFrame(sentences, columns=['text'])
        df_sentences = self.predict_labels_df(df_sentences)

        return df_sentences


def implementation_tab():
    st.write("We'll now use everything learned in the model analysis section to implement the final topic and bias classification model. The code is given below.")
    st.code('''
class MediaBiasModel:

    def __init__(self, label='mediabiasmodel'):
        # Initializes a model and loads the media bias dataset.
        self.label='mediabiasmodel'

        # set the embedding model parameters
        self.embedding_model = "text-embedding-ada-002"
        embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
        self.max_tokens = 8000
        self.encoding = tiktoken.get_encoding(embedding_encoding)


    def load_data(self):
        # Loads the dataset required for training the model and remaps the topics and media bias to simplified labels.

        media_bias_map = {
            'HuffPost': 'left',
            'Federalist': 'right',
            'Daily Beast': 'left',
            'Alternet': 'left',
            'Breitbart': 'right',
            'New Yorker': 'left',
            'American Greatness': 'right', # from https://mediabiasfactcheck.com/american-greatness/
            'Daily Caller': 'right',
            'Daily Wire': 'right',
            'Slate': 'left',
            'Reuters': 'center',
            'Hill': 'center', # from https://mediabiasfactcheck.com/the-hill/
            'USA Today': 'left',
            'CNBC': 'left',
            'Yahoo News - Latest News & Headlines': 'left',
            'AP': 'left',
            'Bloomberg': 'left',
            'Fox News': 'right',
            'MSNBC': 'left',
            'Daily Stormer': 'right', # from https://mediabiasfactcheck.com/the-hill/
            'New York Times': 'left'
            }

        new_topics_map = {
        '#metoo':'activism',
        'abortion':'abortion',
        'black lives matter':'activism',
        'blm':'activism',
        'coronavirus':'coronavirus-and-vaccines',
        'elections-2020':'politics',
        'environment':'environment',
        'gender':'socioeconomics',
        'gun control':'gun-control',
        'gun-control':'gun-control',
        'immigration':'immigration',
        'international-politics-and-world-news':'politics',
        'islam':'islam',
        'marriage-equality':'activism',
        'middle-class':'socioeconomics',
        'sport':'sport',
        'student-debt':'socioeconomics',
        'taxes':'socioeconomics',
        'trump-presidency':'politics',
        'universal health care':'universal-health-care',
        'vaccine':'coronavirus-and-vaccines',
        'vaccines':'coronavirus-and-vaccines',
        'white-nationalism':'white-nationalism',
        }

        df = pd.read_csv('../data/sentences_embeddings.csv')
        if 'embedding' in df.columns:
            df["embedding"] = df.embedding.apply(eval).apply(np.array)  # convert string to array
        
        df = df.rename(columns={'topic':'topic_original'})
        df['topic'] = df['topic_original'].map(new_topics_map)
        df['outlet_bias'] = df['outlet'].map(media_bias_map)
        self.df = df

    def transform_data(self, input_df=None):
        #Transforms the sentence data with OpenAI ADA embeddings.
        # read the API KEY from an environment variable and set it in openai
        if input_df is None:
            if 'embedding' in self.df.columns:
                return self.df
            else:
                # omit reviews that are too long to embed
                self.df["n_tokens"] = self.df.text.apply(lambda x: len(self.encoding.encode(x)))
                self.df = self.df[self.df.n_tokens <= self.max_tokens]

                # get embedding
                self.df["embedding"] = self.df.text.apply(lambda x: get_embedding(x, engine=self.embedding_model))
                return self.df
        else:
            input_df["n_tokens"] = input_df.text.apply(lambda x: len(self.encoding.encode(x)))
            input_df = input_df[input_df.n_tokens <= self.max_tokens]

            # get embedding
            input_df["embedding"] = input_df.text.apply(lambda x: get_embedding(x, engine=self.embedding_model))
            return input_df

    
    def get_text_embedding(self, text):
        # Gets the embedding of a single sentence.

        n_tokens = len(self.encoding.encode(text))
        print(n_tokens)
        if n_tokens > self.max_tokens:
            raise ValueError(f'The input text is too long and resulting in {n_tokens} tokens. The maximum number of tokens for this model is 8000.')
        
        return get_embedding(text, engine=self.embedding_model)

    def fit_models(self):
        # Fits the classifiers for topic, bias and outlet bias on the training data.

        self.topic_model = KNeighborsClassifier(n_neighbors=15)
        self.bias_model = LogisticRegression(C=0.0002, penalty='l2')
        self.politics_model = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=300)

        print('Fitting topic classifier...')
        self.topic_model.fit(np.vstack(self.df.embedding.values), self.df.topic)
        print('Fitting bias classifier...')
        self.bias_model.fit(np.vstack(self.df.embedding.values), self.df.label_bias)
        print('Fitting political bias classifier...')
        self.politics_model.fit(np.vstack(self.df.embedding.values), self.df.outlet_bias)
        
        print('Model fitting complete. To save the current classifiers, run model.save().')

    def load_models(self, model_tag):
        # Loads pre-trained topic, bias and political bias models. 
        # Requires a user-specified model tag to correctly identify models.

        self.model_tag = model_tag
        self.topic_model = pickle.load(open(f'model_topic_{model_tag}.pickle', 'rb'))
        self.bias_model = pickle.load(open(f'model_bias_{model_tag}.pickle', 'rb'))
        self.politics_model = pickle.load(open(f'model_politics_{model_tag}.pickle', 'rb'))
        

    def save_models(self, model_tag=None):
        # Saves the model to a pickle file and makes it available for reuse without re-training.

        if model_tag is None:
            # get a datetime tag to assign to model
            model_tag = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")

        self.model_tag = model_tag
        pickle.dump(self.topic_model, open(f'model_topic_{model_tag}.pickle', 'wb'))
        pickle.dump(self.bias_model, open(f'model_bias_{model_tag}.pickle', 'wb'))
        pickle.dump(self.politics_model, open(f'model_politics_{model_tag}.pickle', 'wb'))

    def predict_labels_text(self, sentence):
        # Predicts the topic, bias and outlet bias of a single sentence.

        embedding = np.array(self.get_text_embedding(sentence))
        topic = self.topic_model.predict(embedding.reshape(1, -1))
        bias = self.bias_model.predict(embedding.reshape(1, -1))
        politics = self.politics_model.predict(embedding.reshape(1, -1))

        return topic, bias, politics

    def predict_labels_df(self, input_df):
        # Transforms text into embeddings and predicts the topic, bias and outlet bias of the 'text' column of an input dataframe.

        input_df = self.transform_data(input_df=input_df)
        input_df['topic'] = self.topic_model.predict(np.vstack(input_df.embedding.values))
        input_df['label_bias'] = self.bias_model.predict(np.vstack(input_df.embedding.values))
        input_df['outlet_bias'] = self.politics_model.predict(np.vstack(input_df.embedding.values))

        return input_df

    def analyze_full_article(self, article_text):
        # Splits the input article text into sentences and analyzes each one separately for bias.

        # transform the article into sentence data and get labels
        sentences = article_text.replace('\n','').replace('U.S.','United States').split('.')
        df_sentences = pd.DataFrame(sentences, columns=['text'])
        df_sentences = self.predict_labels_df(df_sentences)

        return df_sentences
    ''')

    st.write("If you have an OpenAI API key, you can paste it here to fit the model in real time!")
    api_key = st.text_input(label='OpenAI API key', value='', key='impapi')
    if len(api_key)>0:
        openai.api_key = api_key
        st.write('Click the button below to fit the model to the training data!')
        if st.button('Fit model', key='fitmodels'):
            with st.spinner('Initializing model...'):
                model = MediaBiasModel()
            with st.spinner('Loading data...'):
                model.load_data()
            with st.spinner('Transforming data...'):
                model.transform_data()
            with st.spinner('Fitting classifiers...'):
                model.fit_models()

            model_tag = st.text_input("Provide a model tag for saving your models that you can use to load them later.", value="model tag", max_chars=20)
            if st.button('Save models', key='savemodels'):
                model.save_models(model_tag=model_tag)