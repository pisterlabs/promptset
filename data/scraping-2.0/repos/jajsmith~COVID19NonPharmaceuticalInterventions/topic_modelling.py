# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.critical)

# LDA
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess

# Stopwords
import nltk
from nltk.corpus import stopwords

# Regex
import re

# Lemmatization
import spacy
# import fr_core_news_sm

# Printing model topics
from pprint import pprint

# Model Visualization
import pyLDAvis
import pyLDAvis.gensim

# For plotting coherence values
import matplotlib.pyplot as plt

# From source_scraping.py
from source_scraping import load_province

# Dates
from datetime import datetime

# Utils
import numpy as np
import pandas as pd
from warnings import warn

# Preprocessing

def remove_stopwords(texts, stop_words):
    """
    Parameters:
        - `texts`
            a list of documents
        - `stop_words`
            a list of words to be removed from each document in `texts`
    
    Returns: a list of documents that does not contain any element of `stop_words`
    """
    return [[word for word in doc if word not in stop_words] for doc in texts]

def clean_(doc):
    """
    Parameters:
        - `doc`
            a list of words
    
    Returns: a documents with new lines and consecutive spaces replaced by single spaces
    """
    new_doc = [re.sub(r'\s+', ' ', word) for word in doc]
    # More filters if necessary
    return new_doc

def clean(texts):
    """
    Parameters:
        - `texts`
            a list of documents broken into words
    
    Returns: a list of documents with new lines and consecutive spaces replaced by single spaces
    """
    return [clean_(doc) for doc in texts]

def texts_to_words(texts):
    """
    Parameters:
        - `texts`
            a list of documents
    
    Yields: a document broken into words and with punctuation removed
    """
    for doc in texts:
        yield gensim.utils.simple_preprocess(doc, deacc=True)

def lemmatize(texts, allowed_postags=['NOUN', 'ADJ', 'VERB'], lang='english'):
    """
    Parameters:
        - `texts`
            a list of documents broken into words
        - `allowed_postags`
            a list of the parts of speech to be preserved (e.g ['NOUN', 'ADJ'])
        - `lang`
            the language in which the document is written
    
    Returns: a list of documents with words replaced by their lemmas and removed if they do not constitute a part of speech indicated in `allowed_postags`
    """
    
    if lang != 'english':
        warn('Support only currently exists for English language processing')
        return None

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # if lang == 'english' else fr_core_news_sm.load(disable=['parser', 'ner'])
    return [[token.lemma_ for token in nlp(" ".join(doc)) if token.pos_ in allowed_postags] for doc in texts]

def make_bigrams(texts, min_count=5, threshold=100):
    """
    Parameters:
        - `texts`
            a list of documents broken into words
    
    Returns: a list of documents with words arranged into bigrams where applicable
    """
    bigram = Phrases(texts, min_count=min_count, threshold=threshold)
    bigram_mod = Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]
    
# def make_trigrams(texts):
#     return make_bigrams(make_bigrams(texts)) I'm not sure if this one behaves correctly

def custom_preprocess(texts, stop_words, allowed_postags, bigrams=True, lang='english'):
    """
    Parameters:
        - `texts`
            a list of documents broken into words
        - `stop_words`
            a list of words to be removed from each document in `texts`
        - `allowed_postags`
            a list of the parts of speech to be preserved (e.g ['NOUN', 'ADJ'])
        - `bigrams`
            a boolean indicating whether or not to form bigrams from the words in the texts
        - `lang`
            the language in which the texts are written

    
    Returns: a list of preprocessed documents. A result of calling the functions `text_to_words()`, `clean()`, (optionally) `make_bigrams()`, and `lemmatize()` in succession.
    """
    words = list(texts_to_words(texts))
    cleaned_words = clean(words)
    optional_bigrams = make_bigrams(cleaned_words) if bigrams else cleaned_words # Bigrams only if indicated
    return remove_stopwords(lemmatize(optional_bigrams, allowed_postags=allowed_postags, lang=lang), stop_words)

def form_corpus(texts, id2word):
    return [id2word.doc2bow(text) for text in texts]

def dict_corpus(texts):
    """
    Parameters:
        - `texts`
            a list of documents broken into words
    
    Returns: a dictionary and corpus for use in LDA models
    """
    id2word = corpora.Dictionary(texts)
    corpus = form_corpus(texts, id2word)
    return id2word, corpus

# Tuning num_topics hyperparameter

def find_best_model_log_perp(n_topic_range, texts, id2word, corpus, threshold=None, random_state=42, plot=True, verbose=False):
    """
    Searches for the best model in a given range by log perplexity value

    Parameters:
    - `n_topic_range`
        a range of values for the `num_topics` parameter of a gensim LDA model to try
    - `texts`
        a list of documents broken into words
    - `id2word`
        a dictionary containing word encodings
    - `corpus`
        the result of mapping each word in `texts` to its value in `id2word`
    - `random_state` 
        a random state for use in a gensim LDA model
    - `threshold`
        a float that specifies a log perplexity value that if reached will cause the function to return early
    - `plot`
        a boolean specifying whether or not to plot log perplexity values against each `num_topics` value
    - `verbose`
        a boolean specifying whether or not to print updates

    Returns: a tuple containing the best model, the list of all models attempted, and a list of all log perplexity values obtained, respectively.
    """

    models = []
    perp_vals = []

    for n_topics in n_topic_range:

        # Print percentage progress
        if verbose:
            diff = max(n_topic_range) - n_topic_range.start
            print(str(round(100 * (n_topics - n_topic_range.start) / diff, 1)) + "% done")

        lda_model = LdaModel(corpus=corpus,
                            id2word=id2word,
                            num_topics=n_topics,
                            random_state=random_state,
                            update_every=1,
                            chunksize=100,
                            passes=10,
                            alpha='auto',
                            per_word_topics=True
                            )

        p = lda_model.log_perplexity(corpus)
                        
        models.append(lda_model)
        perp_vals.append(p)

    if threshold is not None and p < threshold:
        if verbose: 
            print('Returning early with a log perplexity value of ' + str(p))

        if plot:
            actual_range = range(n_topic_range.start, n_topics + n_topic_range.step, n_topic_range.step)
            plt.plot(actual_range, perp_vals, 'b')
            plt.show()

        return lda_model, models, perp_vals


    if plot:
        # The portion of the range that was actually iterated through
        plt.plot(n_topic_range, perp_vals, 'b')
        plt.show()
    
    return models[np.argmin(perp_vals)], models, perp_vals

def find_best_model_cv(n_topic_range, texts, id2word, corpus, threshold=None, random_state=42, plot=True, verbose=False):
    """
    Searches for the best model in a given range by C_v coherence value

    Parameters:
        - `n_topic_range`
            a range of values for the `num_topics` parameter of a gensim LDA model to try
        - `texts`
            a list of documents broken into words
        - `id2word`
            a dictionary containing word encodings
        - `corpus`
            the result of mapping each word in `texts` to its value in `id2word`
        - `random_state` 
            a random state for use in a gensim LDA model
        - `threshold`
            a float that specifies a coherence value that if reached will cause the function to return early
        - `plot`
            a boolean specifying whether or not to plot coherence values against each `num_topics` value
        - `verbose`
            a boolean specifying whether or not to print updates
    
    Returns: a tuple containing the best model, the list of all models attempted, and a list of all coherence values obtained, respectively.
    """
    models = []
    coherence_vals = []
        
    for n_topics in n_topic_range:
        
        # Print percentage progress
        if verbose:
            diff = max(n_topic_range) - n_topic_range.start
            print(str(round(100 * (n_topics - n_topic_range.start) / diff, 1)) + "% done")
        
        lda_model = LdaModel(corpus=corpus,
                            id2word=id2word,
                            num_topics=n_topics,
                            random_state=random_state,
                            update_every=1,
                            chunksize=100,
                            passes=10,
                            alpha='auto',
                            per_word_topics=True
                            )
        co_model = CoherenceModel(lda_model, texts=texts, dictionary=id2word, coherence="c_v")
        coherence = co_model.get_coherence()
                
        models.append(lda_model)
        coherence_vals.append(coherence)

        if threshold is not None and coherence > threshold:
            if verbose: 
                print('Returning early with a coherence value of ' + str(coherence))

            if plot:
                actual_range = range(n_topic_range.start, n_topics + n_topic_range.step, n_topic_range.step)
                plt.plot(actual_range, coherence_vals, 'b')
                plt.show()

            return lda_model, models, coherence_vals


    if plot:
        # The portion of the range that was actually iterated through
        plt.plot(n_topic_range, coherence_vals, 'b')
        plt.show()
    
    return models[np.argmax(coherence_vals)], models, coherence_vals

# Model visualization

def visualize_model(model, corpus, id2word):
    """
    Parameters:
        - `model`
            a gensim LDA model
        - `corpus`
            the corpus on which the model was trained
        - `id2word`
            the dictionary on which the model was trained
    
    Returns: a pyLDAvis visualization
    """
    pyLDAvis.enable_notebook()
    return pyLDAvis.gensim.prepare(model, corpus, id2word, mds='mmds')

# Pipeline for creation of an LDA model

def lda_from_list(ls, lang='english', bigram=True, allowed_postags=['NOUN', 'VERB', 'ADJ'], stopword_extensions=[], n_topic_range=range(2, 40, 3), threshold=None, use_coherence=True, random_state=42, plot=True, verbose=False):
    """
    Parameters:
    - `list`
        the list containing the documents that will be fed into the LDA model
    - `lang`
        the language in which the documents are written
    - `bigram`
        a boolean indicating whether or not to form bigrams from the words in the texts
    - `allowed_postags`
        a list of the parts of speech to be preserved (e.g ['NOUN', 'ADJ'])
    - `stopword_extensions`
        a list of words to append to the stop words that are removed during preprocessing
    - `n_topic_range`
        a range of values for the `num_topics` parameter of a gensim LDA model to try
    - `threshold`
        a float that specifies a coherence (or log perplexity, see `use_coherence`) value that if reached will cause the function to return early
    - `use_coherence`
        a boolean specifying whether to use coherence as the metric through which the best LDA model is chosen. By default, it is True. When false, the log perplexity is used instead.
    - `random_state`
        a random state for use in a gensim LDA model
    - `plot`
        a boolean specifying whether or not to plot coherence (or log perplexity, see `use_coherence`) values against each `num_topics` value
    - `verbose`
        a boolean specifying whether or not to print updates
    
    Returns:
        a dictionary containing the best model, full model list, list of coherence values, the id2word dictionary, corpus, and texts
    """
    
    find_best_model = find_best_model_cv if use_coherence else find_best_model_log_perp
    texts = ls
    
    if verbose: print("\nPreprocessing Texts\n")
    
    stop_words = stopwords.words(lang)
    stop_words.extend(stopword_extensions)
    texts = custom_preprocess(texts, allowed_postags=allowed_postags, stop_words=stop_words, bigrams=True, lang=lang)
    
    id2word, corpus = dict_corpus(texts)
    
    if verbose: print("\nFinding Best n_topics Values\n")
    model, model_list, co_vals = find_best_model(n_topic_range=n_topic_range, 
        texts=texts, 
        id2word=id2word, 
        corpus=corpus, 
        random_state=random_state, 
        threshold=threshold, 
        plot=plot, 
        verbose=verbose)
    
    return {
        'best_model' : model, 
        'model_list' : model_list, 
        'coherence_vals' : co_vals, 
        'id2word' : id2word, 
        'corpus' : corpus,
        'texts' : texts
    }

def lda_from_df(df, doc_attrib='source_full_text', lang='english', bigram=True, allowed_postags=['NOUN', 'VERB', 'ADJ'], stopword_extensions=[], n_topic_range=range(2, 40, 3), threshold=None, use_coherence=True, random_state=42, plot=True, verbose=False):
    """
    Parameters:
    - `df`
        the dataframe containing the documents that will be fed into the LDA model
    - `doc_attrib`
        the column of the dataframe that contains the documents to be fed into the LDA model
    - `lang`
        the language in which the documents are written
    - `bigram`
        a boolean indicating whether or not to form bigrams from the words in the texts
    - `allowed_postags`
        a list of the parts of speech to be preserved (e.g ['NOUN', 'ADJ'])
    - `stopword_extensions`
        a list of words to append to the stop words that are removed during preprocessing
    - `n_topic_range`
        a range of values for the `num_topics` parameter of a gensim LDA model to try
    - `threshold`
        a float that specifies a coherence (or log perplexity, see `use_coherence`) value that if reached will cause the function to return early
    - `use_coherence`
        a boolean specifying whether to use coherence as the metric through which the best LDA model is chosen. By default, it is True. When false, the log perplexity is used instead.
    - `random_state`
        a random state for use in a gensim LDA model
    - `plot`
        a boolean specifying whether or not to plot coherence (or log perplexity, see `use_coherence`) values against each `num_topics` value
    - `verbose`
        a boolean specifying whether or not to print updates
    
    Returns:
        a dictionary containing the best model, full model list, list of coherence values, the id2word dictionary, corpus, and texts
    """
    
    return lda_from_list(df[doc_attrib], 
        lang=lang,
        bigram=bigram,
        allowed_postags=allowed_postags,
        stopword_extensions=stopword_extensions,
        n_topic_range=n_topic_range,
        threshold=threshold,
        use_coherence=use_coherence,
        random_state=random_state,
        plot=plot,
        verbose=verbose)

def lda_from_province(province, doc_attrib='source_full_text', start_date=datetime(2020, 1, 1), end_date=datetime.today(), bigram=True, allowed_postags=['NOUN', 'VERB', 'ADJ'], stopword_extensions=[], n_topic_range=range(2, 40, 3), threshold=None, use_coherence=True, random_state=42, plot=True, verbose=False):
    """
    Parameters:
    - `province`
        the name of the province on whose news releases the LDA model should be trained
    - `doc_attrib`
        the column of the dataframe that contains the documents to be fed into the LDA model
    - `start_date`
        the date of the earliest news release to be retrieved
    - `end_date`
        the date of the most recent news release to be retrieved
    - `bigram`
        a boolean indicating whether or not to form bigrams from the words in the texts
    - `allowed_postags`
        a list of the parts of speech to be preserved (e.g ['NOUN', 'ADJ'])
    - `stopword_extensions`
        a list of words to append to the stop words that are removed during preprocessing
    - `n_topic_range`
        a range of values for the `num_topics` parameter of a gensim LDA model to try
    - `threshold`
        a float that specifies a coherence (or log perplexity, see `use_coherence`) value that if reached will cause the function to return early
    - `use_coherence`
        a boolean specifying whether to use coherence as the metric through which the best LDA model is chosen. By default, it is True. When False, the log perplexity is used instead
    - `random_state`
        a random state for use in a gensim LDA model
    - `plot`
        a boolean specifying whether or not to plot coherence (or log perplexity, see `use_coherence`) values against each `num_topics` value
    - `verbose`
        a boolean specifying whether or not to print updates
    
    Returns:
        a dictionary containing the best model, full model list, list of coherence values, the id2word dictionary, corpus, and texts
    """
    
    lang = 'french' if province.lower() == 'quebec' else 'english'
    
    df = load_province(province.lower(), before=end_date, verbose=verbose).dropna(subset=[doc_attrib])
    
    # Filter within date range
    df = df[pd.to_datetime(df['start_date']) > start_date]
    df = df[pd.to_datetime(df['start_date']) < end_date]
    
    return lda_from_df(df, 
        doc_attrib=doc_attrib, 
        lang=lang, 
        bigram=bigram, 
        allowed_postags=allowed_postags, 
        stopword_extensions=stopword_extensions,
        n_topic_range=n_topic_range, 
        threshold=threshold,
        use_coherence=use_coherence,
        random_state=random_state,
        plot=plot,  
        verbose=verbose)
