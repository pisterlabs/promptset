#Imports
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
######################################FUNCTIONS######################################
#vectorize(df) //dataframe -> vectorized_text //Array
#get_Cv(model,df_column) //object,array -> coherance  //Decimal
#get_n_topic(df,vectorized_text) //dataframe,array -> n_topic //Integer
#build_lda(n,vectorized_text) //Integer,array -> lda_model //Object
#get_topics(lda_model) //object -> none (with plots) //null
#get_dominant_topics(lda_model,df) //object,dataframe -> topics //Dataframe count of each topic in df
#model_evaluation(lda_model,df) //object,dataframe -> accuracy //Decimal
######################################################################################
#Vectorize text 
def vectorize(df):
    vectorizer = TfidfVectorizer(lowercase=True,
                             analyzer='word',
                             stop_words = 'english',
                             ngram_range = (1,1),
                             max_df = 0.75,
                             min_df = 50,
                             max_features=10000)
    vectorized_text = vectorizer.fit_transform(df['processed_text'])
    return vectorized_text
    
#Getting Coherance Scores
def get_Cv(model, df_columnm):
  topics = model.components_
  n_top_words = 20
  texts = [[word for word in doc.split()] for doc in df_columnm]
  # create the dictionary
  dictionary = corpora.Dictionary(texts)
  # Create a gensim dictionary from the word count matrix
  # Create a gensim corpus from the word count matrix
  corpus = [dictionary.doc2bow(text) for text in texts]
  feature_names = [dictionary[i] for i in range(len(dictionary))]
  # Get the top words for each topic from the components_ attribute
  top_words = []
  for topic in topics:
      top_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
  coherence_model = CoherenceModel(topics=top_words, texts=texts, dictionary=dictionary, coherence='c_v')
  coherence = coherence_model.get_coherence()
  return coherence

#Return the ideal number of topics
def get_n_topic(df,vectorized_text):
    cv_scores = [] #Vector of cv scores
    lg_scores = [] #loglikelihood scores
    perp_scores = [] #Perplexity scores
    for i in range(1,11):
        #Building LDA model
        lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,evaluate_every=-1, 
                                              learning_decay=0.7,learning_method='online', 
                                              learning_offset=50,max_doc_update_iter=100, 
                                              max_iter=20, mean_change_tol=0.001,n_components=i, n_jobs=-1, 
                                              perp_tol=0.1,random_state=20, topic_word_prior=None,total_samples=1000000.0, 
                                              verbose=0)

        lda_topics = lda_model.fit_transform(vectorized_text)
    
        cv_scores.append(get_Cv(lda_model,df['processed_text']))
        lg_scores.append(lda_model.score(vectorized_text))
        perp_scores.append(lda_model.perplexity(vectorized_text))
    max_cv_score = max(cv_scores)
    n_topic = cv_scores.index(max(cv_scores)) + 1
    return n_topic, max_cv_score  #0 index + 1

def build_lda(n,vectorized_text): # n is the ideal number of topics
    lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,evaluate_every=-1, 
                                          learning_decay=0.7,learning_method='online', 
                                          learning_offset=50,max_doc_update_iter=100, max_iter=20, 
                                          mean_change_tol=0.001,n_components=n, n_jobs=-1, perp_tol=0.1,
                                          random_state=20, topic_word_prior=None,total_samples=1000000.0, verbose=0)

    lda_topics = lda_model.fit_transform(vectorized_text)
    #lda_model.components_.shape
    return lda_model   

def get_topics(lda_model,df):
    vectorizer = TfidfVectorizer(lowercase=True,
                             analyzer='word',
                             stop_words = 'english',
                             ngram_range = (1,1),
                             max_df = 0.75,
                             min_df = 50,
                             max_features=10000)
    vectorized_text = vectorizer.fit_transform(df['processed_text'])
    vocab = vectorizer.get_feature_names_out()
    for i, comp in enumerate(lda_model.components_):
        vocab_comp = zip(vocab, comp)
        sorted_words = sorted(vocab_comp, key = lambda x:x[1], reverse = True)
        value_key = [i[0] for i in sorted_words]
        value_count = [i[1] for i in sorted_words]
        plt.rcParams['figure.figsize'] = [25, 5]
        plt.bar(value_key[:10],value_count[:10],align='edge') #Plot top 10 words in each topic
        #plt.figure()
        print("Topic " + str(i) + ": ")
        print(" ".join([i[0] for i in sorted_words[:10]]))
    return

def get_dominant_topics(lda_model,df):
    vectorized_text = vectorize(df)
    lda_topics = lda_model.fit_transform(vectorized_text)
    dominant_topic_list = [np.where(topic == np.max(topic))[0][0] for topic in lda_topics]
    df['dominant_topic'] = dominant_topic_list
    topics = df.dominant_topic.value_counts().sort_index()
    return topics

def get_topic_weights(lda_model,df):
    vectorized_text = vectorize(df)
    lda_topics = lda_model.fit_transform(vectorized_text)
    colnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    docnames = ["Doc" + str(i) for i in range(len(df['processed_text']))]
    df_doc_topic = pd.DataFrame(np.round(lda_topics,2),columns=colnames,index=docnames)
    significant_topic = np.argmax(df_doc_topic.values,axis=1)
    df_doc_topic['dominant_topic'] = significant_topic 
    sns.displot(df_doc_topic['Topic0'].values)
    sns.displot(df_doc_topic['Topic1'].values)
    sns.displot(df_doc_topic['Topic2'].values)
    return df_doc_topic

#######################################Evaluation###############################
#SVM Evaluation
def model_evaluation(lda_model,df):
    vectorized_text = vectorize(df) 
    lda_topics = lda_model.fit_transform(vectorized_text)
    dominant_topic_list = [np.where(topic == np.max(topic))[0][0] for topic in lda_topics]
    df['dominant_topic'] = dominant_topic_list
    x_train, x_test, y_train, y_test = train_test_split(lda_topics, df['dominant_topic'], test_size = 0.2, random_state = 1)
    pd.DataFrame(y_train).value_counts()
    svc = SVC(C=1.0, random_state=1, kernel='poly')
    svc.fit(x_train, y_train)
    x_test_predicted = svc.predict(x_test)
    #np.sum(x_test_predicted == y_test)
    accuracy = (np.sum(x_test_predicted == y_test)/y_test.shape[0]) *100
    return accuracy