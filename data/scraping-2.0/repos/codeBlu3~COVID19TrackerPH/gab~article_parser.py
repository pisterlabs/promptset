# Packages

import pandas as pd
import nltk
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
import glob
from dateutil import parser
import re
from geotext import GeoText
import numpy as np
from scipy import stats 
import geopandas as gpd
import datetime

# Spacy for tokenizing our texts

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Gensim is needed for modeling

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.utils import tokenize
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric

#--------------------------------------------------------------------------------------------

def get_covid_counts():
    
    # Internal Functions -----------------------------------------------------------------
    
    # Setting up Spacy Tokenizer
    nlp = English()

    def lemmatizer(doc):
        # This takes in a doc of tokens from the NER and lemmatizes them. 
        # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
        doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
        doc = u' '.join(doc)
        return nlp.make_doc(doc)

    def remove_stopwords(doc):
        # This will remove stopwords and punctuation.
        # Use token.text to return strings, which we'll need for Gensim.
        doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
        return doc

    # This will add pipelines in our tokenization process.

    nlp.add_pipe(lemmatizer,name='lemmatizer')
    nlp.add_pipe(remove_stopwords, name="stopwords", last=True)
    
    # This is a function that will create a model that predicts the topics conveyed by each group 

    def topic_modeler(tokenized_texts, no_topics, no_words):
        topics = []

        words = corpora.Dictionary(tokenized_texts)
        corpus = [words.doc2bow(doc) for doc in tokenized_texts]

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=words,
                                                    random_state = 3,
                                                   num_topics= no_topics)

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_texts, dictionary=words, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        return lda_model
    
    #-------------------------------------------------------------------------------------
    
    # Getting the data -----------------------------
    
    files = glob.glob("scraped_data\*.csv")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs,ignore_index=True)
    df['date'] = [parser.parse(date).strftime('%Y-%m-%d') for date in df['date']]
    #df = df[(df['text'].str.contains('coronavirus'))]
    #df = df[df['category'] == 'Philippines']
    df = df.drop_duplicates()
    df = df.reset_index(drop = True)


    location = pd.read_csv('ph_locations.csv')
    location = location.applymap(str.lower)
    
    
    # Get LDA Topics -------------------------------

    words = df['text'].str.lower()
    listWords = []
    for item in words:
        listWords.append([nlp(item)])

    topics = []
    for x in listWords:
        res = topic_modeler(x, 1, 10)
        res = res.show_topic(0, topn = 10)
        topics.append([word[0] for word in res])

    df['LDA_Topics'] = topics
    
    # Extracting all the counting phrases in the articles -------------------------

    df['count_docs'] =  df['text'].apply(lambda x: re.findall("\d+(?:,\d+)?\s+[a-zA-Z]+", x))

    checker = ['confirmed','suspected','quarantine','case','infected','monitoring','chinese','monitored']

    count_docs = []
    for index, row in df.iterrows():
        passed = []
        for item in row['count_docs']:
            if any(ext in item.lower() for ext in checker):
                passed.append(item)
                break

        count_docs.append(passed)

    df['count_docs'] = count_docs
    
    # Extracting all the PH Locations using geotext on the articles ----------------------------

    df['PH_Loc'] = [list(set(GeoText(content, 'PH').cities)) for content in df['text']]
    df['PH_Loc'] = [[x.lower() for x in w] for w in df['PH_Loc']]
    df['PH_Loc'] =[[x.replace('city', '') for x in w] for w in df['PH_Loc']]
    
    
    # Identifying which articles are about suspicious or confirmed cases of the virus ---------------------------
    
    status = []
    for index, row in df.iterrows():
        if (('confirmed' in row['LDA_Topics']) | ('confirms' in row['LDA_Topics'])| ('confirm' in row['LDA_Topics'])) & ('confirm' in row['title'])  & (row['date'] >= '2020-01-30'):
            status.append('confirmed')
        elif (('confirmed' in row['LDA_Topics']) | ('confirms' in row['LDA_Topics'])| ('confirm' in row['LDA_Topics']))& (row['date'] >= '2020-01-30'):
            status.append('confirmed')
        elif ('confirm' in row['title']) & ('coronavirus' in row['title']) & (row['date'] >= '2020-01-30'):
            status.append('confirmed')
        elif (any(words in row['LDA_Topics']  for words in ['suspected','quarantine','case','infected','monitoring']))& ('FACT CHECK' not in row['title']) & ('FALSE' not in row['title']):
            status.append('suspected')
        else:
            status.append('')
    df['status'] = status
    
    # Selecting Provinces in the identified locations -------------------------------------------------------

    df['PH_Loc'] = [list(set(loc) & set(location['Pro_Name'].unique())) for loc in df['PH_Loc']]

    # For locations not identified through the text, it will check with the LDA topics if a location is identified and use it instead

    for index, row in df.iterrows():
        if len(row['PH_Loc']) == 0:
            try:
                df.loc[index, 'PH_Loc'] = [list(set(row['LDA_Topics']) & set(location['Pro_Name'].unique()))]
            except ValueError:
                continue
                
    # Cleaning the document counts to just numbers ------------------------------------

    counts = []
    case = []
    for count in df['count_docs']:
        try:
            counts.append(count[0].split(' ')[0])
        except IndexError:
            counts.append(0)

        try:
            case.append(count[0].split(' ')[1])
        except IndexError:
            case.append('')

    df['counts'] = counts
    df['counts'] = [str(count).replace(',', '') for count in df['counts']]
    df['counts'] = [str(count).replace('.', '') for count in df['counts']]
    df['case'] = case
    
    # Finalizing Locations ----------------------------------------------------

    ph_loc = []
    for loc in df['PH_Loc']:
        try:
            ph_loc.append(loc[0])
        except IndexError:
            ph_loc.append('')
    df['Loc'] = ph_loc
    
    # Fixing confirmed counts -----------------------------------------------

    count_fixer = []
    for index, row in df.iterrows():
        if (row['status'] == 'confirmed') & (row['case'] != 'confirmed'):
            count_fixer.append(1)
        else:
            count_fixer.append(row['counts'])

    df['counts'] = count_fixer
    
    df.to_csv('pre_processed_data.csv', index = False)
    
    # Filtering tables to those with PH Locations identified -----------------------
    
    df = df[df['counts'] != '1300\n\nConfirmed']
    df = df[df['Loc'] != '']
    
    df['counts'] = df.counts.astype(int)
    df = df.sort_values('date')
    
    # Parse the results ------------------------------------------------------
    def parse(df):
        #print(df.info())

        # Get min/max/mean values
        dfa = pd.pivot_table(df, values = 'counts', index=['date', 'Loc'], columns='status', aggfunc=[min, max, np.mean, stats.mode])

        # Remove multi-index
        dfa.columns = ["_".join(pair) for pair in dfa.columns]
        dfa = dfa.reset_index()

        # Replace 0 with np.nan to forward fill null values
        dfa = dfa.replace(0, np.nan)

        # Forward filling needs to be by area
        places = list(df['Loc'].unique())

        global dfb
        dfb = pd.DataFrame()
        for place in places:
            df_temp = dfa[dfa['Loc'] == place].fillna(method='ffill')
            dfb = dfb.append(df_temp)
        return dfb
    
    res = parse(df)
    
    # Cleaning results -----------------------------------------------------
    
    res = res[['date','Loc', 'min_suspected','min_confirmed']]
    res = res.fillna(0)
    res = res.sort_values('date')
    
    # Completing running total per date -------------------------------------
    
    def add_row(df, row):
        df.loc[-1] = row
        df.index = df.index + 1  
        return df.sort_index()

    for Loc in res['Loc']:
        sus_holder = 0
        con_holder = 0
        for date in res['date']:
            if sum(res[res['date'] == date]['Loc'].str.contains(Loc)) > 0:

                sus_holder = res[(res['date'] == date) & (res['Loc'] == Loc)]['min_suspected'].iloc[0]
                con_holder = res[(res['date'] == date) & (res['Loc'] == Loc)]['min_confirmed'].iloc[0]
            else:
                add_row(res, [date, Loc, sus_holder, con_holder])

    
    res = res.sort_values('date')
    
    # Including the geolocations of each location -------------------------------------
    
    prov = gpd.read_file('prov_shp/prov_geo.shp')
    df = pd.merge(res, prov, left_on = 'Loc', right_on = 'Pro_Name')
    
    df['lat'] = [cor.split(',')[0] for cor in df['centroid']]
    df['long'] = [cor.split(',')[1] for cor in df['centroid']]
    
    # Cleaning final results -------------------------------------------------------
    
    df = df[['date','Loc','min_suspected','min_confirmed','long','lat']]
    df.columns = (['Date','Location','Suspected','Confirmed','Longitude','Latitude'])

    df['Date'] = [datetime.datetime.strptime(str(date), '%Y-%m-%d').strftime('%Y-%m-%dT%H:%M:%S.%f') for date in df['Date']]
    
    df = df.sort_values('Date')
    
    df.to_csv('ncov_parsed.csv', index = False)
    return df
