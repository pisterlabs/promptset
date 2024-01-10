import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy import stats
import re
from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import ldamodel
# from gensim.test.utils import common_texts

def make_csv(df):
    df = pd.read_excel('globalterrorismdb_0718dist.xlsx')
    df.to_csv('globalterrorismdb_0718dist.csv', encoding='utf-8', index=False)

def clean_gtd():
    df = pd.read_csv('data/globalterrorismdb_0718dist.csv', low_memory=False)

    '''military and police and primary target types for suicide bombings'''
    '''create binary variable where military/police == 1 and all else == 0'''
    df['mil_pol_targ'] = np.where(((df['targtype1'] == 3) | (df['targtype1'] == 4)), 1, 0)

    '''military and police and primary target types for suicide bombings'''
    '''create binary variable where military/police == 1 and all else == 0'''
    df['mil_check'] = np.where((df['targsubtype1'] == 36), 1, 0)
    df['mil_barr'] = np.where((df['targsubtype1'] == 27), 1, 0)
    df['pol_check'] = np.where((df['targsubtype1'] == 24), 1, 0)
    df['pol_build'] = np.where((df['targsubtype1'] == 22), 1, 0)
    df['rel_place'] = np.where((df['targsubtype1'] == 86), 1, 0)
    df['util_elec'] = np.where((df['targsubtype1'] == 107), 1, 0)
    df['gov_polit'] = np.where((df['targsubtype1'] == 15), 1, 0)
    df['terr_nonstate'] = np.where((df['targsubtype1'] == 94), 1, 0)
    df['mil_check'] = np.where((df['targsubtype1'] == 36), 1, 0)

    '''Middle East + NA are primary region of suicide bombings'''
    '''create binary variable where ME and NA == 1 and all else == 0'''
    df['ME_NA'] = np.where((df['region'] == 10), 1, 0)

    '''Bombing/Explosion is primary means of attack for suicide bombings'''
    '''create binary variable where B/E == 1 and all else == 0'''
    df['bomb_explo'] = np.where((df['attacktype1'] == 3), 1, 0)

    '''Different types of attacks'''
    '''create binary variable where attack subtype == 1 and all else == 0'''
    df['explo_vehicle'] = np.where((df['weapsubtype1'] == 15), 1, 0)
    df['explo_unknown'] = np.where((df['weapsubtype1'] == 16), 1, 0)
    df['firearm_unknown'] = np.where((df['weapsubtype1'] == 5), 1, 0)
    df['firearm_rifle'] = np.where((df['weapsubtype1'] == 2), 1, 0)
    df['explo_project'] = np.where((df['weapsubtype1'] == 11), 1, 0)
    df['explo_other'] = np.where((df['weapsubtype1'] == 17), 1, 0)
    df['firearm_handgun'] = np.where((df['weapsubtype1'] == 3), 1, 0)

    '''code -9 values as missing so that there are only 1 and 0 in this variable'''
    df['claimed'].replace(to_replace=[-9],value=np.NaN, inplace=True)

    '''create binary variables for modes of claiming reponsibilty'''
    df['claim_internet'] = np.where((df['claimmode'] == 7), 1, 0)
    df['claim_note'] = np.where((df['claimmode'] == 5), 1, 0)
    df['claim_personal'] = np.where((df['claimmode'] == 8), 1, 0)

    '''create binary variable where year >= 2003 == 1 and all else == 0'''
    df['year_2003'] = np.where((df['iyear'] >=2003), 1, 0)

    '''create binary variables for countries known to have suicide bombings'''
    df['Iraq'] = np.where((df['country'] == 95), 1, 0)
    df['Afghanistan'] = np.where((df['country'] == 4), 1, 0)
    df['India'] = np.where((df['country'] == 92), 1, 0)
    df['Columbia'] = np.where((df['country'] == 45), 1, 0)
    df['Syria'] = np.where((df['country'] == 200), 1, 0)

    df_suicide_DT=df[['ME_NA', 'claimed', 'year_2003', 'mil_barr', 'pol_check', 'pol_build', 'rel_place', 'util_elec', 'gov_polit', 'terr_nonstate', 'mil_check', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'firearm_rifle', 'explo_project', 'explo_other', 'firearm_handgun', 'claim_internet', 'claim_note', 'claim_personal', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'motive', 'suicide']]
    df_suicide_DT.to_csv('data/df_suicide_DT.csv', index=False)

def up_down_sample():
    df = pd.read_csv('data/df_suicide_DT.csv', low_memory=False)
    pd.options.display.max_columns = 200
    df.dropna(inplace = True)

    '''the data are imbalanced -- only 5% of suicide == 1'''
    '''discussion: https://stats.stackexchange.com/questions/28029/training-a-decision-tree-against-unbalanced-data'''
    '''here is how to deal with it: https://elitedatascience.com/imbalanced-classes'''
    df_majority = df[df.suicide==0]
    df_minority = df[df.suicide==1]

    df_majority_downsampled = resample(df_majority, replace=False, n_samples=2569, random_state=123)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    # print(df_downsampled['suicide'].value_counts())

    df_minority_upsampled = resample(df_minority, replace=True, n_samples=46374, random_state=123)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    # print(df_upsampled['suicide'].value_counts())

    create_test_train(column_clean(df_downsampled, filename='data/df_suicide_downsampled.csv'), filename1='data/df_suicide_downsampled_Train.csv', filename2='data/df_suicide_downsampled_Test.csv')
    create_test_train(column_clean(df_upsampled, filename='data/df_suicide_upsampled.csv'), filename1='data/df_suicide_upsampled_Train.csv', filename2='data/df_suicide_upsampled_Test.csv')

def column_clean(df, filename):
    df['ME_NA'] = df['ME_NA'] == True
    df['claimed'] = df['claimed'] == True
    df['suicide'] = df['suicide'] == True
    df['year_2003'] = df['year_2003'] == True

    df['mil_check'] = df['mil_check'] == True
    df['mil_barr'] = df['mil_barr'] == True
    df['pol_check'] = df['pol_check'] == True
    df['pol_build'] = df['pol_build'] == True
    df['rel_place'] = df['rel_place'] == True
    df['util_elec'] = df['util_elec'] == True
    df['gov_polit'] = df['gov_polit'] == True
    df['terr_nonstate'] = df['terr_nonstate'] == True
    df['mil_check'] = df['mil_check'] == True

    df['explo_vehicle'] = df['explo_vehicle'] == True
    df['explo_unknown'] = df['explo_unknown'] == True
    df['firearm_unknown'] = df['firearm_unknown'] == True
    df['firearm_rifle'] = df['firearm_rifle'] == True
    df['explo_project'] = df['explo_project'] == True
    df['explo_other'] = df['explo_other'] == True
    df['firearm_handgun'] = df['firearm_handgun'] == True
    df['claim_internet'] = df['claim_internet'] == True
    df['claim_note'] = df['claim_note'] == True
    df['claim_personal'] = df['claim_personal'] == True

    df['ishostkid'] = df['ishostkid'] == True

    df['Iraq'] = df['Iraq'] == True
    df['Afghanistan'] = df['Afghanistan'] == True
    df['India'] = df['India'] == True
    df['Columbia'] = df['Columbia'] == True
    df['Syria'] = df['Syria'] == True

    df=df.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)

    # df.to_csv(filename, index=False)
    return df

def create_test_train(df, filename1, filename2):
    y = df.pop('suicide').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)

    X_train = pd.DataFrame(data=X_train, index = None, columns = ['ME_NA', 'claimed', 'year_2003', 'mil_barr', 'pol_check', 'pol_build', 'rel_place', 'util_elec', 'gov_polit', 'terr_nonstate', 'mil_check', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'firearm_rifle', 'explo_project', 'explo_other', 'firearm_handgun', 'claim_internet', 'claim_note', 'claim_personal', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'motive'])
    X_test = pd.DataFrame(data=X_test, index = None, columns = ['ME_NA', 'claimed', 'year_2003', 'mil_barr', 'pol_check', 'pol_build', 'rel_place', 'util_elec', 'gov_polit', 'terr_nonstate', 'mil_check', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'firearm_rifle', 'explo_project', 'explo_other', 'firearm_handgun', 'claim_internet', 'claim_note', 'claim_personal', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'motive'])
    y_train = pd.DataFrame(data=y_train, index = None, columns = ['suicide'])
    y_test = pd.DataFrame(data=y_test, index = None, columns = ['suicide'])

    Train = pd.concat([X_train, y_train], axis=1)
    Test = pd.concat([X_test,y_test], axis=1)
    Train.to_csv(filename1, index=False)
    Test.to_csv(filename2, index=False)
    print (Train.shape, Test.shape)

def make_LDA(df, filename):
    df.reset_index(inplace=True)

    df['motive'].replace(to_replace=['Unknown', 'The specific motive for the attack is unknown.'],value=np.NaN, inplace=True)
    data = df[['motive']]
    data.dropna(inplace = True)
    data_text = data[['motive']]
    documents = data_text

    stemmer = SnowballStemmer('english')
    processed_docs = documents['motive'].map(preprocess)
    texts = [[''.join(item) for item in document] for document in processed_docs]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    np.random.seed(1) # setting random seed to get the same results each time.
    model = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
    topics = model.show_topics()

    doc_top_prob = [model.get_document_topics(corpus[i]) for i in range(len(corpus))]

    '''THERE ARE SOMETIMES MISSING DATA POINTS IN PART OF A TUPLE IN THE UPSAMPLE. THIS REMOVES THOSE CASES'''

    three_scores = []
    for i in doc_top_prob:
        if len(i) == 3:
            three_scores.append(i)
        else:
            pass

    topic1_tup = []
    topic2_tup = []
    topic3_tup = []
    for i, j, k in three_scores:
        topic1_tup.append(i)
        topic2_tup.append(j)
        topic3_tup.append(k)

    topic1 = []
    topic2 = []
    topic3 = []
    for i in topic1_tup:
        topic1.append(i[1])
    for i in topic2_tup:
        topic2.append(i[1])
    for i in topic3_tup:
        topic3.append(i[1])

    df_topic1 = pd.DataFrame(i for i in topic1)
    df_topic2 = pd.DataFrame(i for i in topic2)
    df_topic3 = pd.DataFrame(i for i in topic3)

    df_processed_docs = pd.DataFrame(data=processed_docs)
    df_processed_docs.reset_index(inplace=True)
    final = pd.concat([df_processed_docs,df_topic1,df_topic2,df_topic3], axis=1)
    final.columns = ['index', 'motive2', 'topic1', 'topic2', 'topic3']
    LDA_merged = df.merge(final, how='right', on='index')

    '''make ready for Decision Tree'''
    LDA_merged = LDA_merged.drop(['motive', 'motive2'], axis=1)
    LDA_merged.to_csv(filename, index=False)

def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    my_stopwords = {'motive', 'attack', 'Unknown', 'unknown', 'however', 'sources', 'specific', 'stated', 'statement', 'States', 'state', 'target', 'speculate', 'incident', 'targeted', 'targeting', 'speculated', 'suicide', 'bomb', 'bombing', 'bomber', 'responsibility', 'claim', 'claimed', 'noted', 'State', 'carried', 'majority', 'minority'}
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in my_stopwords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
