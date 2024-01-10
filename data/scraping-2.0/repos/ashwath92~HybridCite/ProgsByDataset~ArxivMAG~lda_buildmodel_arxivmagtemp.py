""" Trains an LDA Mallet model on the Arxiv data set (with bigrams, lemmatization)."""
from gensim import corpora, models, similarities
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.wrappers import LdaMallet
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing import preprocessing
from gensim.utils import simple_preprocess, tokenize, has_pattern
import contractions
import os
import re
from tqdm import tqdm
from pprint import pprint
import pickle
import spacy
#import nltk
#from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer
snowball = SnowballStemmer(language='english')

# disable parse, named entity recognition to speed it up.
nlp = spacy.load('en', disable=['parser', 'ner'])

# Add new stop words: https://stackoverflow.com/questions/41170726/add-remove-stop-words-with-spacy
# |= : syntactic sugar for creating doing a union with the  with {}
nlp.Defaults.stop_words |= {'table', 'ref', 'formula', 'citation', 'cit', 'references'
                            'fig', 'figure', 'abstract', 'introduction',
                            'description','conclusion','results','discussion'}

# Load the Mallet LDA Java  program
mallet_path = '/home/ashwath/mallet-2.0.8/bin/mallet'
# Initialize Word Net 
#lemmatizer = WordNetLemmatizer()

def back_to_string(texts):
    """ Generator which yields string versions of the sub-lists """
    new_texts=[]
    for text in texts:
        new_texts.append(" ".join(text))
    return new_texts 


def lemmatize2(content, allowed_tags=re.compile(r'(NN|VB|JJ|RB)'), light=False,
              stopwords=frozenset(), min_length=2, max_length=15):
    """Use the English lemmatizer from `pattern <https://github.com/clips/pattern>`_ to extract UTF8-encoded tokens in
    their base form aka lemma, e.g. "are, is, being" becomes "be" etc.
    This is a smarter version of stemming, taking word context into account.
    MODIFIED to return only wordsm and not concatenated tags!
"""
    if not has_pattern():
        raise ImportError(
            "Pattern library is not installed. Pattern library is needed in order to use lemmatize function"
        )
    from pattern.en import parse
    if light:
        import warnings
        warnings.warn("The light flag is no longer supported by pattern.")
    # tokenization in `pattern` is weird; it gets thrown off by non-letters,
    # producing '==relate/VBN' or '**/NN'... try to preprocess the text a little
    # FIXME this throws away all fancy parsing cues, including sentence structure,
    # abbreviations etc.
    content = u' '.join(tokenize(content, lower=True, errors='ignore'))
    parsed = parse(content, lemmata=True, collapse=False)
    result = []
    for sentence in parsed:
        for token, tag, _, _, lemma in sentence:
            if min_length <= len(lemma) <= max_length and not lemma.startswith('_') and lemma not in stopwords:
                if allowed_tags.match(tag):
                    #lemma += "/" + tag[:2]
                    result.append(lemma.encode('utf8'))
    return result

'''
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
import thinc.extra.datasets
import plac
import spacy
from spacy.util import minibatch

allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']
def lemmatization(nlp, batch_id, texts):
    """https://spacy.io/api/annotation"""
    print("Lemmatizing using Spacy")
    texts_gen = back_to_string(texts)
    texts_out = []
    for doc in nlp.pipe(texts_gen):#, n_threads=64, batch_size=10):
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

partitions = minibatch(test, size=10)
executor = Parallel(n_jobs=64, backend="multiprocessing", prefer="processes")
do = delayed(partial(lemmatization, nlp))
tasks = (do(i, batch) for i, batch in enumerate(partitions))
executor(tasks)
'''
'''def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']):
    """https://spacy.io/api/annotation"""
    print("Lemmatizing using Spacy")
    texts_gen = back_to_string(texts)
    texts_out = []
    #for row, doc in enumerate(nlp.pipe(texts_gen, n_threads=64, batch_size=10000)):
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
# https://nlpforhackers.io/topic-modeling/
# https://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html
# https://www.machinelearningplus.com/nlp/gensim-tutorial/
'''
'''
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
'''
'''
def lemmatization(texts):
    """Use nltk and Wordnet to lemmatize."""
    print("Lemmatizing using pattern/gensim")
    #texts_gen = back_to_string(texts)
    texts_out = []
    # KEEP ONLY NOUNS, ADJ, VERB, ADV
    for sent in tqdm(texts):
        texts_out.append([porter.stem(word) for word in sent])#lemmatize2(sent))
    return texts_out
'''
def snowballstem(texts):
    """Use nltk and Snowball stemmer to stem."""
    print("Stemming using Snowball Stemmer")
    #texts_gen = back_to_string(texts)
    texts_out = []
    # KEEP ONLY NOUNS, ADJ, VERB, ADV
    for sent in tqdm(texts):
        texts_out.append([snowball.stem(word) for word in sent])#lemmatize2(sent))
    return texts_out
'''
def lemmatization(texts):
    """Use gensim and pattern to lemmatize, function lemmatize modified to lemmatize2 (tags removed)."""
    print("Lemmatizing using pattern/gensim")
    texts_sen = back_to_string(texts)
    texts_out = []
    # KEEP ONLY NOUNS, ADJ, VERB, ADV
    #for row, doc in enumerate(nlp.pipe(texts_gen, n_threads=64, batch_size=10000)):
    for sent in tqdm(texts_sen):
        texts_out.append(lemmatize2(sent))
    del texts_sen
    return texts_out
'''
def make_bigrams(texts):
    print("Making bigrams")
    return [bigram_mod[doc] for doc in tqdm(texts)]

def remove_stopwords(texts):
    print("Removing stop words.")
    return [[word for word in simple_preprocess(str(doc)) if word not in nlp.Defaults.stop_words] for doc in tqdm(texts)]

def clean_text(text):
    """ Cleans the text in the only argument in various steps. NOT USED. 
    ARGUMENTS: text: content/title, string
    RETURNS: cleaned text, string"""
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Remove stop words
    text = preprocessing.remove_stopwords(text)
    # Remove html tags
    text = preprocessing.strip_tags(text)
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(preprocessing.strip_punctuation(text))
    return text

def stream_from_file(filename):
    """ Generator function to stream from the input file. Memory-friendly"""
    with open(filename, 'r') as file:
        for line in file:
            yield clean_text(line).split()

class InputCorpus:
    """ Class to generate the collection of texts in a numerical form. Memory-efficient"""
    def __init__(self, input_file, dictionary):
        """
        Yield each document in turn, as a list of tokens (unicode strings).
        
        """
        self.input_file = input_file
        self.dictionary = dictionary
        
    def __iter__(self):
        """ Each yield statement yields lists (generators) of tuples of (word_id, count),..."""
        for tokens in stream_from_file(filename):
            yield self.dictionary.doc2bow(tokens)

filename = '/home/ashwath/Programs/ArxivCS/arxiv_hd2v_training.txt'
# memory-hungry
print("Starting")
try:
    with open('processedtext.pickle', 'rb') as ipick:
        data_stemmed = pickle.load(ipick)
except FileNotFoundError:
    #with open(filename, 'r') as file:
        # list of lists
        # Remove punctuation at this stage
    #    data = [simple_preprocess(line, deacc=True, min_len=2) for line in tqdm(file)]
        # REMOVE THIS PICKLE LATER!!!!!!
    with open('inputtext.pickle', 'rb') as dpick:
        data = pickle.load(dpick)
    print("Pickle loaded")
    # Create a Phrases model
    
    bigram = Phrases(data, min_count=10, threshold=100) # higher threshold fewer phrases.
    bigram_mod = Phraser(bigram)

    data_nostops = remove_stopwords(data)
    #Form Bigrams
    data_bigrams = make_bigrams(data_nostops)
    # Change lemmatization from spacy to pattern
    #data_lemmatized = lemmatization(data_bigrams)#, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN'])
    
    # Use a Snowball stemmer, lemmatization takes too much time and CPU
    data_stemmed = snowballstem(data_bigrams)
    print("Stemmed, bigrams added, stop words removed, time to pickle the list")
    with open('processedtext.pickle', 'wb') as opick:
        pickle.dump(data_stemmed, opick)
    # Save some memory
    del data
    del data_nostops
    del data_bigrams

try:
    id2word_dictionary = corpora.Dictionary.load('arxivmag.dict')
except FileNotFoundError:
    id2word_dictionary = corpora.Dictionary(data_stemmed)
    id2word_dictionary.save('arxivmag.dict')  # save dict to disk
    print("id2word dictionary created")

# Trim the dictionary: it has 2019397 tokens.
# Discard tokens which occur in less than 5 documents, discard those which occur in more than 20%
# of the documents, and then keep the 500000 most common words of what is left
#id2word_dictionary.filter_extremes(no_below=5, no_above=0.2, keep_n=500000)
try:
    corpus = corpora.MmCorpus('arxivmag_bow_corpus.mm')
except FileNotFoundError:
    corpus = [id2word_dictionary.doc2bow(textlist) for textlist in tqdm(data_stemmed)]
    print("Doc2Bow corpus created")
    # TOO BIG TO SERIALIZE
    # Save the Dict and Corpus
    try:
        corpora.MmCorpus.serialize('arxivmag_bow_corpus.mm', corpus)  # save corpus to disk
    except OverflowError:
        # Don't save corpus, call LDA directly
        print("Overflow while saving corpus, skip and train.")
        ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=500, id2word=id2word_dictionary)
        print('LDA Model trained')

        try:
            lda_model.save('lda_model.model')
        except OverflowError:
            print("Trying to pickle model using protocol 4")
            with open('lda_model.model', 'wb') as pick:
                pick.dump(lda_model, pick, protocol=pickle.HIGHEST_PROTOCOL)
        print("Lda model saved to disk")

        # Show Topics
        pprint(ldamallet.show_topics(formatted=False))

        # Compute Coherence Score
        coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_stemmed,
                                           dictionary=id2word_dictionary, coherence='c_v')

        coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        print('\nCoherence Score: ', coherence_ldamallet)


# Memory-friendly
# Create generator 
# docstream is a generator which will be passed to Dictionary to create a Gensim dictionary
# docstream = (tokens for tokens in stream_from_file(filename))

# Dictionary takes a list of list of tokens. it associates each word (value) to a numeric id (key)
# id2word_dictionary = corpora.Dictionary(docstream)
# create a stream of bag-of-words vectors from the dictionary, we have to iterate through the file again
# arxiv_corpus = InputCorpus(filename, id2word_dictionary)

# Build the LDA model: don't use tfidf: https://stackoverflow.com/questions/44781047/necessary-to-apply-tf-idf-to-new-documents-in-gensim-lda-model
#lda_model = models.LdaMulticore(corpus=corpus, num_topics=500, id2word=id2word_dictionary, workers=63, random_state=13)
#lda_model = models.LdaModel(corpus=corpus, num_topics=200, id2word=id2word_dictionary)

ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=500, id2word=id2word_dictionary)
print('LDA Model trained')

try:
    lda_model.save('lda_model.model')
except OverflowError:
    print("Trying to pickle model using protocol 4")
    with open('lda_model.model', 'wb') as pick:
        pick.dump(lda_model, pick, protocol=pickle.HIGHEST_PROTOCOL)

print("Lda model saved to disk")

# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_stemmed,
                                           dictionary=id2word_dictionary, coherence='c_v')

coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)
# # Load them back
#loaded_dict = corpora.Dictionary.load('mydict.dict')

#corpus = corpora.MmCorpus('bow_corpus.mm')
#for line in corpus:
#    print(line)