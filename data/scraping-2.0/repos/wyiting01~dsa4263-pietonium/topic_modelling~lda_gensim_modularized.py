import pandas as pd
import gensim
import gensim.corpora as corpora 
from gensim import models
from gensim.models import CoherenceModel
from nltk import FreqDist
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

def read_data(path):
    """
    Read cleaned data
    """
    processed_data = pd.read_csv(path)
    return processed_data

def split_sentence(text):
    """
    Split a sentence into a list 
    """
    lst = []
    for sentence in text:
        lst.append(sentence.split())
    return lst

def get_top_10_words(data_inv):
    """
    Get the top 10 highest count words across all reviews
    """
    lst = []
    for i in range(len(data_inv)):
        for k in data_inv[i]:
            lst.append(k)
    fdist = FreqDist(lst) # a frequency distribution of words (word count over the corpus)
    top_k_words, _ = zip(*fdist.most_common(10)) # unzip the words and word count tuples
    return(list(top_k_words)) 

def remove_top_10_words(data_inv, top_10_lst):
    """
    Remove the top 10 highest count words from all reviews
    """
    lst = []
    for i in range(len(data_inv)):
        sentence = []
        for k in data_inv[i]:
            if k not in top_10_lst:
                sentence.append(k)
        lst.append(sentence)
    return lst

def preprocess_words(data_inv):
    """
    1. Split sentences to individual words
    2. Removed top 10 highest frequency count words
    2. Build bigram and trigram models
    3. Form bigram and trigram
    4. Create dictionary
    5. Create corpus
    """
    data_words = split_sentence(data_inv)
    
    remove_lst = get_top_10_words(data_words)
    data_words = remove_top_10_words(data_words,remove_lst)

    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)  
    
    data_bigrams = make_bigrams(data_words, bigram_phrases)
    data_bigrams_trigrams = make_trigrams(data_bigrams, bigram_phrases, trigram_phrases)

    id2words = corpora.Dictionary(data_bigrams_trigrams)

    collection_texts = data_bigrams_trigrams
    bow_corpuss = [id2words.doc2bow(text) for text in collection_texts]

    return ([data_bigrams_trigrams, id2words, bow_corpuss])

def make_bigrams(texts, bigram_text):
    """
    Faster way to detect bigrams
    """
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_text, trigram_text,):
    """
    Faster way to detect bigrams/trigrams
    """
    bigram_mod = gensim.models.phrases.Phraser(bigram_text)
    trigram_mod = gensim.models.phrases.Phraser(trigram_text)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def combine_reviews_to_list(input_data):
    """
    Combine all processed text into a list
    """
    data = input_data.processed_text.values.tolist()
    return data

def obtain_corpus(data):
    """
    Obtain preprocessed text, dictionary and bag of words corpus
    """
    data_words, id2word, bow_corpus = preprocess_words(data)
    return (data_words, id2word, bow_corpus)

def get_tfidf_corpus(corpus):
    """
    Create tfidf corpus from bag of words corpus
    """
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf

def load_base_lda_model(data_words, id2word, corpus_tfidf):
    """
    1. Run and return Base model 
    2. Print Baseline coherence score
    """
    lda_tfidf_model_baseline = gensim.models.LdaMulticore(corpus=corpus_tfidf, 
                                                 num_topics=2, 
                                                 id2word=id2word, 
                                                 random_state=100,
                                                 passes=10,
                                                 workers=4)
    coherence_model_lda = CoherenceModel(model=lda_tfidf_model_baseline, 
                                        texts = data_words, 
                                        dictionary=id2word,
                                        coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Baseline Coherence Score: ', coherence_lda)
    return lda_tfidf_model_baseline

def get_coherence_values_and_optimal_topic_num(type_corpus, n, id2word, data_words):
    """
    1. Compute coherence values of different n --- n is the number of topics
    2. Obtain optimal number of topics
    3. Show plot of coherence values against n
    """
    topics = []
    score = []
    flag = True
    optimal_topic_index = 0
    optimal_topic_no = 0

    # optimal number of topics start from 2 
    # one topic is "ignored"
    for i in range(2, n):
        lda_model = gensim.models.LdaMulticore(corpus=type_corpus,
                                               id2word=id2word,
                                                num_topics=i, 
                                                random_state=100,
                                                chunksize=100,
                                                passes=10,
                                                workers=4)
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
    plt.show()
    return [optimal_topic_no, score[optimal_topic_index], df_output]

def load_final_lda_model(id2word, corpus_tfidf, optimal_topic_no):
    """
    Run and return Final LDA model 
    """    
    # final model with parameters yielding highest coherence score
    lda_tfidf_model = gensim.models.LdaMulticore(corpus=corpus_tfidf,
                                        id2word=id2word,
                                        num_topics=optimal_topic_no,
                                        random_state=100,
                                        passes=10,
                                        workers=4)
    return lda_tfidf_model

def dominant_topic_per_review(chosen_model, corpus, texts):
    """
    Find the dominant topic per review
    """
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(chosen_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
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


def topic_distri_across_review(df):
    """
    Number of Reviews in Each Topic
    """
    topic_counts = df['Dominant_Topic'].value_counts()
    return topic_counts

def unique_keyword_per_topic(lda_model):
    """
    To find the unique sets of keywords in each topic
    """
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

def unique_sets(lda_model):
    """
    Print out the unique sets of keywords in each topic
    """
    unique_sets = unique_keyword_per_topic(lda_model)
    for i in range (len(unique_sets)):
        print('Topic {}: {}'.format(i, unique_sets[i]))

def get_all_topic_distribution(chosen_model, corpus):
    """
    Finding the topic percent contribution for each review
    """
    sent_topics_df = pd.DataFrame()
    topic0_prob = []
    topic1_prob = []
    topic2_prob = []
    dominant_topic = []
    for i, row in enumerate(chosen_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
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
    sns.displot(sent_topics_df['Topic1'].values).set(title='Topic 1')
    sns.displot(sent_topics_df['Topic2'].values).set(title='Topic 2')
    plt.show()
    return sent_topics_df