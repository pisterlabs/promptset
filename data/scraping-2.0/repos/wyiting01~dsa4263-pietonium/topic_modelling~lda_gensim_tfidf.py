# LDA model with Tfidf
# adapted from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# adapted from https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
# adapted from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
# adapted from https://github.com/wjbmattingly/topic_modeling_textbook/blob/main/03_03_lda_model_demo_bigrams_trigrams.ipynb
import pandas as pd
import gensim
import gensim.corpora as corpora 
from gensim import models
import pyLDAvis.gensim
import pickle 
import pyLDAvis
import os
from pathlib import Path
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from nltk import FreqDist
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

# split a sentence into a list 
def split_sentence(text):
    lst = []
    for sentence in text:
        lst.append(sentence.split())
    return lst

# get the top 10 highest count words across all reviews
def get_top_10_words(data_inv):
    lst = []
    for i in range(len(data_inv)):
        for k in data_inv[i]:
            lst.append(k)
    fdist = FreqDist(lst) # a frequency distribution of words (word count over the corpus)
    top_k_words, _ = zip(*fdist.most_common(10)) # unzip the words and word count tuples
    return(list(top_k_words)) 

# remove the top 10 highest count words from all reviews
def remove_top_10_words(data_inv, top_10_lst):
    lst = []
    for i in range(len(data_inv)):
        sentence = []
        for k in data_inv[i]:
            if k not in top_10_lst:
                sentence.append(k)
        lst.append(sentence)
    return lst

def preprocess_words(data_inv):
    # split sentences to individual words
    data_words = split_sentence(data_inv)
    remove_lst = get_top_10_words(data_words)
    data_words = remove_top_10_words(data_words,remove_lst)

    # Build the bigram and trigram models
    # bigrams are two words frequently occur together
    # trigrams are three words frequently occurring 
    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)  

    # form bigrams
    data_bigrams = make_bigrams(data_words, bigram_phrases)

    # form trigrams
    data_bigrams_trigrams = make_trigrams(data_bigrams, bigram_phrases, trigram_phrases)

    # Create dictionary 
    id2words = corpora.Dictionary(data_bigrams_trigrams)

    # Create Corpus
    collection_texts = data_bigrams_trigrams

    # Term Document Frequency
    # convert list of words into bag-of-words format
    bow_corpuss = [id2words.doc2bow(text) for text in collection_texts]

    return ([data_bigrams_trigrams, id2words, bow_corpuss])

def make_bigrams(texts, bigram_text):
    # faster way to detect bigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_text, trigram_text,):
    # faster way to detect bigrams/trigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    trigram_mod = gensim.models.phrases.Phraser(trigram_text)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# compute coherence values of different k
def compute_coherence_values(type_corpus, k, id2word, data_words):
    topics = []
    score = []
    flag = True
    optimal_topic_index = 0
    optimal_topic_no = 0

    # optimal number of topics start from 2 
    # one topic is "ignored"
    for i in range(2, k):
        lda_model = gensim.models.LdaMulticore(corpus=type_corpus,
                                               id2word=id2word,
                                                num_topics=i, 
                                                random_state=100,
                                                chunksize=100,
                                                passes=10)
        # instantiate topic coherence model
        cm = CoherenceModel(model=lda_model, texts= data_words, dictionary=id2word, coherence='c_v')
        # Append number of topics modeled
        topics.append(i)
        # Append coherence scores to list
        score.append(cm.get_coherence()) 
        # check for highest cm before flattening
        if i != 2 and flag == True:
            if cm.get_coherence() < score[i-3]:
                optimal_topic_no = i - 1
                optimal_topic_index = i - 3
                flag = False

    # print the coherence score of topic number from 2 to k
    for m, cv in zip(topics, score):  
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
    # save the coherence score and topic number as dataframe
    output = {'k': topics, 'coherent score': score}
    df_output = pd.DataFrame(output)
    
    # plot the graph of the coherence score and topic number 
    plt.plot(df_output['k'], df_output['coherent score'])
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.title("Coherence score with respective number of topics")
    plt.savefig('./result/coherence_plot.png', dpi=300, bbox_inches='tight')
    #plt.show()

    return [optimal_topic_no, score[optimal_topic_index], df_output]

# Finding the dominant topic for each review
def dominant_topic_per_review(chosen_model, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each review
    for i, row in enumerate(chosen_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = chosen_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return(df_dominant_topic)
    
# Topic distribution across reviews
def topic_distri_across_review(dorminant_topic_each_sent):
    # Number of Reviews for Each Topic
    topic_counts = dorminant_topic_each_sent['Dominant_Topic'].value_counts()
    return topic_counts
   
# to find the unique sets for each topic
def unique_keyword_per_topic(lda_model):
    # convert the print_topics result to dictionary format
    for idx, topic in lda_model.print_topics(num_words=10):    
        print('Topic: {} \nWords: {}'.format(idx, topic))
    my_dict = {'Topic_' + str(i): [token for token, score in lda_model.show_topic(i, topn=10)] for i in range(0, lda_model.num_topics)}
    topics_keywords = []
    for key,value in my_dict.items():
        topics_keywords.append(set(value))
    # find the intersection between the 3 topics
    result = topics_keywords[0].intersection(topics_keywords[1],topics_keywords[2])
    final_unique_set = []
    for i in range(len(topics_keywords)):
        for j in range(len(topics_keywords)):
            if j != i:
                for k in range(len(topics_keywords)):
                    if k not in [j,i]:
                        set_unique = (topics_keywords[i]^result^topics_keywords[j]^topics_keywords[k])&topics_keywords[i]
                        if set_unique not in final_unique_set:
                            final_unique_set.append(set_unique)
                        break
                break
    return (final_unique_set)

def get_tfidf_corpus(bow_corpus):
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf

# Finding the topic percent contribution for each review
def get_all_topic_distribution(chosen_model, corpus):
    # Init output
    sent_topics_df = pd.DataFrame()
    topic0_prob = []
    topic1_prob = []
    topic2_prob = []
    dominant_topic = []
    # Get main topic in each review
    for i, row in enumerate(chosen_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                dominant_topic.append(topic_num)
            if int(topic_num) == 0:
                topic0_prob.append(round(prop_topic,4))
            elif int(topic_num) == 1:
                topic1_prob.append(round(prop_topic,4))
            else:
                topic2_prob.append(round(prop_topic,4))
    sent_topics_df['Topic0'] = topic0_prob
    sent_topics_df['Topic1'] = topic1_prob
    sent_topics_df['Topic2'] = topic2_prob
    sent_topics_df['Dominant Topic'] = dominant_topic
    sns.displot(sent_topics_df['Topic0'].values).set(title='Topic 0')
    plt.savefig('./result/topic0_distribution.png', dpi=300, bbox_inches='tight')
    sns.displot(sent_topics_df['Topic1'].values).set(title='Topic 1')
    plt.savefig('./result/topic1_distribution.png', dpi=300, bbox_inches='tight')
    sns.displot(sent_topics_df['Topic2'].values).set(title='Topic 2')
    plt.savefig('./result/topic2_distribution.png', dpi=300, bbox_inches='tight')
    #plt.show()
    return sent_topics_df

if __name__ == '__main__':
    
    # read in post processed data
    processed_data = pd.read_csv('../data/curated/reviews/cleaned_reviews.csv')

    # combine all the processed text into a list
    data = processed_data.processed_text.values.tolist()

    # preprocess words
    data_words, id2word, bow_corpus = preprocess_words(data)
    # Removed these words: ('taste', 'like', 'good', 'great', 'product', 'flavor', 'make', 'one', 'get', 'use') (3207, 2369, 1855, 1798, 1753, 1753, 1508, 1415, 1405, 1351)
    
    # create tfidf corpus
    corpus_tfidf = get_tfidf_corpus(bow_corpus)

    # number of topics for baseline
    num_topics = 2

    # Running LDA using TF-IDF corpus
    lda_model_tfidf = gensim.models.LdaMulticore(corpus=corpus_tfidf, 
                                                 num_topics=num_topics, 
                                                 id2word=id2word, 
                                                 random_state=100,
                                                 passes=10,
                                                 workers=4)

    # baseline model coherence
    coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts = data_words, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Baseline Coherence Score: ', coherence_lda)
    
    # create directory to keep models
    os.makedirs('../model/lda_gensim/', exist_ok=True)

    # save Baseline model
    lda_model_tfidf.save('../model/lda_gensim/lda_tfidf_model_baseline.pkl')
    
    # create directory to keep results
    os.makedirs('result/', exist_ok=True)
    
    # hyperparameter tuning on number of topics
    final_num_topics, final_score, coherence_score_topic = compute_coherence_values(corpus_tfidf,10, id2word, data_words)
    coherence_score_topic.to_csv('./result/coherence_score_topic.csv')
    print('Final Coherence Score:', final_score)
    print('Final number of topics used:', final_num_topics)

    # final model with parameters yielding highest coherence score
    final_lda_model_tfidf = gensim.models.LdaMulticore(corpus=corpus_tfidf,
                                        id2word=id2word,
                                        num_topics=final_num_topics,
                                        random_state=100,
                                        passes=10,
                                        workers=4)
    
    final_lda_model_tfidf.save('../model/lda_gensim/lda_tfidf_model_FINAL.pkl') 

    # Print the Keyword for each topic
    for idx, topic in final_lda_model_tfidf.print_topics(num_words=10):    
        print('Topic: {} \nWords: {}'.format(idx, topic))
    
    # load model
    lda_tfidf_model = gensim.models.LdaMulticore.load('../model/lda_gensim/lda_tfidf_model_FINAL.pkl')
    
    # Visualize the topics
    LDAvis_data_filepath = os.path.join('../model/lda_gensim/kl_ldavis_tfidf_'+str(final_num_topics)+'.pkl')
    filePath = Path(LDAvis_data_filepath)
    filePath.touch(exist_ok= True)
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_tfidf_model, corpus_tfidf, id2word, R = 10)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    #load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, 'result/kl_ldavis_tfidf_'+str(final_num_topics)+'.html')
    LDAvis_prepared
    
    # Find the dominant topic for each review
    df_topic_per_key = dominant_topic_per_review(lda_tfidf_model, corpus_tfidf, data)
    df_topic_per_key.to_csv('./result/dominant_topic_in_each_sentence.csv')
    
    # Topic distribution across documents
    df_dominant_topic = topic_distri_across_review(df_topic_per_key)
    
    # Topic percentage contribution for each review
    topic_perc_dis = get_all_topic_distribution(lda_tfidf_model, corpus_tfidf)
    
    topic_perc_dis.to_csv('./result/topic_perc_dis.csv')

    unique_sets = unique_keyword_per_topic(lda_tfidf_model)
    for i in range (len(unique_sets)):
        print('Topic {}: {}'.format(i, unique_sets[i]))        

    """
    Topic 0: {'much', 'really', 'chocolate', 'cup'} -- dessert
    Topic 1: {'would', 'bar', 'find', 'chip', 'snack'} -- snacks and chips
    Topic 2: {'food', 'cat', 'china', 'dog', 'treat'} -- pet food
    """






