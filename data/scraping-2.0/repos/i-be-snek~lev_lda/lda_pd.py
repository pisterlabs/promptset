from gensim.utils import simple_preprocess # for data format
from gensim.models.phrases import Phrases, Phraser # for making bigrams/trigrams
import gensim.corpora as corpora # for making the corpus
from gensim.models import CoherenceModel, KeyedVectors # for evaluation, loading word embeddings
from gensim.models.ldamodel import LdaModel # for performing LDA
from pprint import pprint 
import logging
from tabulate import tabulate # for displaying topics in markdown
import numpy as np # for generated topics
import pandas as pd # for csv export

""" default parameters """
num_topics = 10 # desired number of topics to generate as output
# For alpha and eta, see gensim doc. https://radimrehurek.com/gensim/models/ldamodel.html
alpha = 0.01 # choose 'auto' to learn alpha from data
eta = 'auto' 
iterations = 10 # default is 50
bigrams = True # False if using unigrams
embeddings = False # False if not using word embeddings
topn = 200 # top n similar words to grab from word embeddings
model_path = 'add_path_to_embeddings'
keywords = ['صحه','مرض','مصاب','وباء','عنايه','علاج','وفيات','طبية','سلامه','موت'] #examples
coherence = True # False outputs no score
data_path = 'example.txt'
output = 'example' 
use_keywords = True # If False, standard LDA is performed and no word embeddings are loaded
logging.basicConfig(filename=output+'.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
output_csv = output + '.csv'


def display_log(str):
    logging.info(str)
    print(str)
# loads word embeddings
def load_model(model_path):
    model = KeyedVectors.load_word2vec_format(model_path)
    display_log("Word embeddings loaded")
    return model

# grabs topn most similar words from word embeddings
# regardless of similarity score
# TODO: grab most similar up to a threshold similarity score
def grab_most_similar(list_keys,model,top):

    main_list = []
    for i in list_keys:
        try:
            tup_list = model.most_similar(i,topn=top+1)
            for x in tup_list:
                main_list.append(x[0])
        except:
            logging.warning("Word {} not in vocab.".format(i))

    return main_list

# appends topn most similar to keyword list
def add_similar(list_keys, most_similar):
    before=len(list_keys)
    list_keys = list_keys + most_similar
    list_keys = set(list_keys)
    display_log("Up to {} new words added to keyword list".format(len(list_keys)-before))
    display_log("New size is {}.".format(len(list_keys)))

    return list_keys

# eliminates documents not containing one or more of the keywords
def word_lists(data_path, keywords):
    data = open(data_path, 'r')
    for tweet in data:
        text = simple_preprocess(str(tweet))
        if any(x in keywords for x in text):
            yield(text)

def word_lists_no_keywords(data):
    data = open(data_path, 'r')
    for tweet in data:
        text = simple_preprocess(str(tweet))
        yield(text)

def make_bigrams(words):
    bigram = Phraser(Phrases(words, min_count=4, threshold=60))
    return [bigram[doc] for doc in words]


def LDA_pd(data=data_path, 
            list_keys=keywords, 
            num_topics=num_topics, 
            iterations=iterations, 
            alpha=alpha, 
            eta=eta, 
            embeddings=embeddings,
            top=topn,
            output_path=output,
            use_keywords=use_keywords):

    output = open(output_path+'.output', 'w')
    output.write("Generating {} topics from {} initial keywords \n".format(num_topics, len(keywords))) 
    output.write("LDA model parameters:\n(1) alpha {}\n(2) eta {}\n(3) running {} iterations. \n".format(alpha, eta, iterations))

    if use_keywords: # if false, LDA is performed on all data (NOT Partial Data LDA)
        data_words = list(word_lists(data, list_keys))
        output.write("Standard set of keywords includes:\n" + ', '.join(i for i in list_keys))
        if embeddings:
            display_log("Loading word embeddings")
            model = load_model(model_path)
            most_similar=grab_most_similar(list_keys,model=model,top=topn)
            list_keys = add_similar(list_keys, most_similar)
            output.write("Supplemented keyword list includes:\n" + ', '.join(i for i in list_keys))
            output.write('\n')
            output.write("Top {} most similar words added from word emeddings (if found) \n".format(topn))
    else:
        data_words = list(word_lists_no_keywords(data_path))

    display_log("Created data word list of size {}".format(str(len(data_words))))

    # generate bigrams
    if bigrams:
        data_words = make_bigrams(data_words)
        display_log("Created bigrams word list")
        output.write("Topic integrates bigrams.\n\n")

    # create dictionary
    id2word = corpora.Dictionary(data_words)
    display_log("Created dictionary")

    # TDF
    corpus = [id2word.doc2bow(text) for text in data_words]
    display_log("Created corpus")

    #LDA model
    lda_model = LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=60,
                                            passes=25,
                                            alpha=alpha,
                                            eta=eta,
                                            iterations=iterations)
    display_log("Created LDA model")
    #pprint(lda_model.print_topics())

    topic_header = ["Topic " + str(i+1) for i in range(num_topics)]
    topic_array = np.array([lda_model.show_topic(i) for i in range(num_topics)]).T
    output.write("Topics\n-----------------------\n")
    output.write(tabulate(topic_array[0], headers=topic_header, tablefmt='github'))
    output.write("\n\n")
    output.write("Similarity Scores\n-----------------------\n")
    output.write(tabulate(topic_array[1], headers=topic_header, tablefmt='github'))
    output.write("\n\n")
    
    display_log("printed table into output file " + output_path) 


    df_all = pd.DataFrame()
    topics_transposed = topic_array.T
    for i in range(num_topics):
        new = pd.DataFrame(topics_transposed[i], columns=['Topic '+str(i),'score'])
        df_all = pd.concat([df_all, new], axis=1)
    df_all.to_csv(output_csv, index=False, encoding='utf-16')
    display_log("Exported topics and scores into csv file " + output_csv+'.csv') 

   
    #coherence for LDA-PDs
    if coherence:
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
    
        output.write("Coherence and Preplexity Scores\n-----------------------\n")
        output.write("LDA-PD Model with {} keywords: \n Perplexity: {} \n Coherence: {}".format(len(keywords),
                                                                    lda_model.log_perplexity(corpus),
                                                                    coherence_lda))
        display_log("Coherence and Perplexity calculated, see " + output_path+'.output')
    
    display_log("Log saved in " + output_path+'.log')
    display_log("Output saved in " + output_path+'.output')
    display_log("Topics saved in " + output_path+'.csv')


    return lda_model

if __name__ == '__main__':
    LDA_pd()