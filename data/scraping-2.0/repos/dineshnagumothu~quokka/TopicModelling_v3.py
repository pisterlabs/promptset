from nltk.corpus import stopwords
import pandas as pd
import re
from pprint import pprint
import gensim
from gensim.utils import simple_preprocess
import spacy
import math
import gensim.corpora as corpora
import os
from gensim.models.wrappers import LdaMallet
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import sys



def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
def remove_stopwords(stop_words,texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(bigram_mod,texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(trigram_mod,bigram_mod, texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(nlp,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def compute_coherence_values(mallet_path,dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print ("Model for "+str(num_topics)+ " number of topics")
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    doc_topics = []
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        topic_probs = []
        for j, (topic_num, prop_topic) in enumerate(row):
          topic_probs.append(prop_topic)
        doc_topics.append(topic_probs)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df, doc_topics)

def corpusProcessing(articles):
    #create a new data frame
    '''df = pd.DataFrame({'id':[],'title': [], 'paragraphs': []})
    #load articles to dataframe
    i=0
    for record in articles:
        paragraph=""
        i+=1
        for p in record['paragraphs']:
            paragraph+=p
        #print (paragraph)
        df=df.append({'id':i,'title':record['title'],'paragraphs':paragraph},ignore_index=True)
    print ("Articles Loaded to Dataframe")'''
        
    #preprocess text data
    data = articles.text.values.tolist()
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    #pprint(data[:1])
    print("Text pre-processing done")
    return data

def plotCoherenece(coherence_values,x):
    #plot coherence value vs number of topics
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    print ("Chart plotted for coherence score vs num. of topics")
def optimalModel(model_list,coherence_values,x):
    #calculate optimal model
    max=0
    i=0;opt_number=0;
    for m, cv in zip(x, coherence_values):
        if max<cv:
            max=cv
            opt_number=i
        i+=1    
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    #print (opt_number)
    print ("Optimum number calculated as", opt_number)
    
    return model_list[opt_number]

def ldatopicmodel(articles):
    data=corpusProcessing(articles)
    #convert sentences to words
    data_words = list(sent_to_words(data))
    #print(data_words[:1])
    print ("Text converted to words")
    
    #converts phrases to bigrams and trigrams
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    #print(trigram_mod[bigram_mod[data_words[0]]])
    print ("Bigram and Trigram models created")
    
    #Take stopwords from NLTK
    stop_words = stopwords.words('english')
    #print (stop_words)
    print ("Stop words loaded")
    
    #remove stopwords, make bigrams and load spacy nlp for lemmatization
    data_words_nostops = remove_stopwords(stop_words, data_words)
    data_words_bigrams = make_bigrams(bigram_mod,data_words_nostops)
    print ("Bigrams generated")
    nlp = spacy.load('en', disable=['parser', 'ner'])
    print ("Spacy package loaded")

    #change spacy nlp settings as per requirements 
    count=0
    for bigrams in data_words_bigrams:
        for words in bigrams:
            count+=len(words)
    count=12583746
    print ("Before"+str(nlp.max_length))
    nlp.max_length=count+1
    print ("After"+str(nlp.max_length))
    print ("Spacy settings adjusted for the data")
    data_lemmatized=lemmatization(nlp,data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #print(data_lemmatized[:1])
    print ("Data Lemmatized")
    
    #form a dictionary of words
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    #print(corpus[:1])
    print ("Corpus Created")
    #print(id2word[0])
    #print ([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
    
    #load mallet path and pass it to compute coherence function
    os.environ['MALLET_HOME'] = '/content/mallet-2.0.8'
    mallet_path = 'mallet-2.0.8/bin/mallet' # update this path
    print (mallet_path)
    print("Mallet Path loaded successfully")
    start=8; limit=16; step=2
    print ("Coherence Value Computation for different number of topics")
    #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)
    model_list, coherence_values = compute_coherence_values(mallet_path=mallet_path,dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=start, limit=limit, step=step)
    print ("Coherence computation done") 
    x = range(start, limit, step)
    plotCoherenece(coherence_values,x)
    optimal_model=optimalModel(model_list,coherence_values,x)
    print ("Optimum model considered")
    model_topics = optimal_model.show_topics(formatted=False)
    #pprint(optimal_model.print_topics(num_words=10))
    
    df_topic_sents_keywords, doc_topics = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
    print("Text adopted for the model")
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return (df_dominant_topic, optimal_model, bigram_mod, trigram_mod, id2word, doc_topics)

def fetch_topics(df, bigram_mod, trigram_mod, id2word, lda_model):
    data=corpusProcessing(df)
    #convert sentences to words
    data_words = list(sent_to_words(data))
    #print(data_words[:1])
    print ("Text converted to words")

    #bigram_mod = gensim.models.phrases.Phraser(bigram)
    #trigram_mod = gensim.models.phrases.Phraser(trigram)
    #print(trigram_mod[bigram_mod[data_words[0]]])
    #print ("Bigram and Trigram models created")
    
    #Take stopwords from NLTK
    stop_words = stopwords.words('english')
    #print (stop_words)
    print ("Stop words loaded")
    
    #remove stopwords, make bigrams and load spacy nlp for lemmatization
    data_words_nostops = remove_stopwords(stop_words, data_words)
    data_words_bigrams = make_bigrams(bigram_mod,data_words_nostops)
    print ("Bigrams generated")
    nlp = spacy.load('en', disable=['parser', 'ner'])
    print ("Spacy package loaded")

    #change spacy nlp settings as per requirements
    count=12583746
    print ("Before"+str(nlp.max_length))
    nlp.max_length=count+1
    print ("After"+str(nlp.max_length))
    print ("Spacy settings adjusted for the data")
    data_lemmatized=lemmatization(nlp,data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #print(data_lemmatized[:1])
    print ("Data Lemmatized")

    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    #print(corpus[:1])
    print ("Corpus Created") 

    df_topic_sents_keywords, doc_topics = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)
    print("Text adopted for the model")
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return (df_dominant_topic, doc_topics)
    

