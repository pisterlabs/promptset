# https://www.kaggle.com/kernels/scriptcontent/11511967/notebook

import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import ast
import re
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
import seaborn as sb

from sklearn.feature_extraction.text import CountVectorizer
# from textblob import TextBlob
import scipy.stats as stats

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

import nltk
from nltk.tag import tnt
from nltk.corpus import stopwords
from nltk.corpus import indian
from nltk.tokenize import word_tokenize,sent_tokenize

import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora

import stanfordnlp

# custom imports
from util import create_str_from_list,diff

import warnings
warnings.filterwarnings("ignore")

# %matplotlib inline

"""
Initialize variables and config 

"""
datafile = './data/hindi.txt'
stanford_model_path = "./models/hi_hdtb_models/"
words_list = []

# stanfordNLP initialization
# config = {
#     "processors" : "tokenize,mwt,lemma,pos",
#     "lang" : "hi",
#     # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
#     'tokenize_model_path': stanford_model_path + '/hi_hdtb_tokenizer.pt', 
# 	'pos_model_path': stanford_model_path + '/hi_hdtb_tagger.pt',
# 	'pos_pretrain_path': stanford_model_path + '/hi_hdtb.pretrain.pt',
# 	'lemma_model_path': stanford_model_path + '/hi_hdtb_lemmatizer.pt'
# }

# nlp = stanfordnlp.Pipeline(**config) # Initialize the pipeline using a configuration dict
nlp = stanfordnlp.Pipeline ( lang = 'hi' ) # use when you are playing with different langauges

# dictionary that contains pos tags and their explanations
pos_dict = {
'CC': 'coordinating conjunction','CD': 'cardinal digit','DT': 'determiner',
'EX': 'existential there (like: \"there is\" ... think of it like \"there exists\")',
'FW': 'foreign word','IN':  'preposition/subordinating conjunction','JJ': 'adjective \'big\'',
'JJR': 'adjective, comparative \'bigger\'','JJS': 'adjective, superlative \'biggest\'',
'LS': 'list marker 1)','MD': 'modal could, will','NN': 'noun, singular \'desk\'',
'NNS': 'noun plural \'desks\'','NNP': 'proper noun, singular \'Harrison\'',
'NNPS': 'proper noun, plural \'Americans\'','PDT': 'predeterminer \'all the kids\'',
'POS': 'possessive ending parent\'s','PRP': 'personal pronoun I, he, she',
'PRP$': 'possessive pronoun my, his, hers','RB': 'adverb very, silently,',
'RBR': 'adverb, comparative better','RBS': 'adverb, superlative best',
'RP': 'particle give up','TO': 'to go \'to\' the store.','UH': 'interjection errrrrrrrm',
'VB': 'verb, base form take','VBD': 'verb, past tense took',
'VBG': 'verb, gerund/present participle taking','VBN': 'verb, past participle taken',
'VBP': 'verb, sing. present, non-3d take','VBZ': 'verb, 3rd person sing. present takes',
'WDT': 'wh-determiner which','WP': 'wh-pronoun who, what','WP$': 'possessive wh-pronoun whose',
'WRB': 'wh-abverb where, when','QF' : 'quantifier, bahut, thoda, kam (Hindi)','VM' : 'main verb',
'PSP' : 'postposition, common in indian langs','DEM' : 'demonstrative, common in indian langs'
}

"""
Functions
"""
# Define helper functions: get top words in the corpus
def get_top_n_words(n_top_words, count_vectorizer, text_data):
    '''
    returns a tuple of the top n words in a sample and their 
    accompanying counts, given a CountVectorizer object and text sample
    '''
    vectorized_headlines = count_vectorizer.fit_transform(text_data)
#     print(vectorized_headlines)
    vectorized_total = np.sum(vectorized_headlines, axis=0)
    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)
    word_values = np.flip(np.sort(vectorized_total)[0,:],1)
    
    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
    for i in range(n_top_words):
        word_vectors[i,word_indices[0,i]] = 1

    words = [word[0] for 
             word in count_vectorizer.inverse_transform(word_vectors)]

    return (words, word_values[0,:n_top_words].tolist()[0])

# Remove Stopwords, Make Bigrams, Lemmatize, tag POS

def remove_stopwords(word_tokenized):
    """
    Inputs:
        words_tokenized: preprocessed, tokenized list of words
    --------------------------------------------
    Returns:
        list: list of the words with stopwords removed
    """
    return [word for word in words_list if word not in stop_words]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# Lemmatization

# returns dataframe
def extract_lemma(doc):
    """
    Inputs:
        doc: the documents of corpus
    --------------------------------------------
    Returns:
        dataframe: word, lemma as columns
    """
    parsed_text = {'word':[], 'lemma':[]}
    for sent in doc.sentences:
        for wrd in sent.words:
#             print(wrd.text)
            #extract text and lemma
            parsed_text['word'].append(wrd.text)
            parsed_text['lemma'].append(wrd.lemma)
    return pd.DataFrame(parsed_text)

# returns list 
def lemmatization(words_tokenized):
    """
    Inputs:
        words_tokenized: preprocessed, tokenized list of words
    --------------------------------------------
    Returns:
        list: lemma of the words
    """
    return [word.lemma for word in words_list] 

# def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent)) 
#         texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     return texts_out


# POS tagger

# returns a dataframe of pos and text
def extract_pos(doc):
    """
    Inputs:
        doc: the documents of corpus
    --------------------------------------------
    Returns:
        dataframe: word, pos, exp as columns
    """
    parsed_text = {'word':[], 'pos':[], 'exp':[]}
    for sent in doc.sentences:
        for wrd in sent.words:
            if wrd.pos in pos_dict.keys():
                pos_exp = pos_dict[wrd.pos]
            else:
                pos_exp = 'NA'
            parsed_text['word'].append(wrd.text)
            parsed_text['pos'].append(wrd.pos)
            parsed_text['exp'].append(pos_exp)
    return pd.DataFrame(parsed_text)

# custom_removal of garbage from the words_list
def custom_remove_garbage(original_words_list,list_of_garbage_words):
    tmp_list = [word for word in original_words_list if word not in list_of_garbage_words] # garbage list
    tmp_list = [word for word in tmp_list if len(re.findall("\d+",word))==0] # english numbers
    tmp_list = [word for word in tmp_list if len(re.findall("[a-zA-Z]+",word))==0] # english alphabets
    return tmp_list

###############################################################################################################################################
# EXECUTES FROM HERE
###############################################################################################################################################1k2
# Read the pre-processed file and split using double full-stops ("॥") and append it to the text:list
# No need of sentence tokenization
with open(datafile,'r',encoding='utf-8') as f:
    text = f.read()
    text = text.split("॥")

# remove the sloka numbers from the txt
text = [word for word in tqdm(text) if not re.findall("^\s?\d+\s?",word)]

# convert the list to df for understanding only
text_df = pd.DataFrame(text)

# hindi stopwords
stop_words_df = pd.read_csv("./data/stopwords.txt", header = None)
stop_words = list(set(stop_words_df.values.reshape(1,-1).tolist()[0]))
stop_words.extend(["।", "।।", ")", "(", ",","","हे" ]) # extend the list so as to remove garbage from the pre-processed data 

# count_vectorizer
count_vectorizer = CountVectorizer(stop_words=stop_words)
count_vectorizer.fit_transform(text)
words, word_values = get_top_n_words(n_top_words=15,
                                     count_vectorizer=count_vectorizer, 
                                     text_data=text)


lemmatized = lemmatization(text)
clean_text = remove_stopwords(lemmatized)

# remove punctuation from each word
table = str.maketrans('', '', string.punctuation)
punctuation_stripped = [w.translate(table) for w in clean_text]

# Difference between the clean_text and stripped
resultant = diff(clean_text,punctuation_stripped)

# remove garbage
final_text = custom_remove_garbage(stripped,resultant)

# word tokenization
for line in text:
    words_list = words_list + word_tokenize(line)

# words_list = [word_tokenize(line) for line in text] # creates list of lists

"""
Create the Dictionary and Corpus needed for Topic Modeling
The two main inputs to the LDA topic model are the dictionary(id2word) and the corpus.
"""
# Create Dictionary
id2word = corpora.Dictionary([final_text])
# Create Corpus
texts = final_text

# Term Document Frequency
corpus = [id2word.doc2bow(texts)] #for text in texts]

"""
Gensim creates a unique id for each word in the document. The produced corpus shown above is a mapping of 
(word_id, word_frequency).
For example, (0, 1) above implies, word id 0 occurs once in the first document. Likewise, word id 1 occurs twice and so on.
"""
# View

# print(corpus[:1])
# print(id2word[3]) # word a given id corresponds to

# Human readable format of corpus (term-frequency)

# [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=20,
                                           alpha='auto',
                                           per_word_topics=True)

"""
Print the Keyword in the 10 topics
It means the top 10 keywords that contribute to this topic are: ‘hockey’, ‘pts_pt’, ‘pit’.. and 
so on and the weight of ‘hockey’ on topic 0 is 0.044.
"""
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

"""
Compute Model Perplexity and Coherence Score
Model perplexity and topic coherence provide a convenient measure to judge how good a given topic model is.
In my experience, topic coherence score, in particular, has been more helpful.
"""
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=final_text, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

"""
Visualize the topics-keywords
pyLDAvis package’s interactive chart is designed to work well with jupyter notebooks.
Each bubble on the left-hand side plot represents a topic. The larger the bubble, the more prevalent is that topic.
A good topic model will have fairly big, non-overlapping bubbles scattered throughout the chart instead of being clustered in one quadrant.
A model with too many topics, will typically have many overlaps, small sized bubbles clustered in one region of the chart.
"""
# Visualize the topics

# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis