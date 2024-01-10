test_text = "In the modern age, computers have revolutionized nearly every aspect of human life, significantly altering the way we work, communicate, and entertain ourselves. While their integration into society brings numerous advantages, it also presents certain challenges. This essay explores the pros and cons of computers in our society. On the positive side, computers have streamlined productivity and efficiency across various industries. They have simplified complex tasks, increasing accuracy and speed. Computers enable seamless communication, connecting people across the globe, fostering collaboration, and broadening cultural exchange. Moreover, they have revolutionized education, providing limitless resources and interactive learning experiences. Another advantage lies in the entertainment realm. Computers offer diverse multimedia experiences, from gaming to virtual reality, enhancing leisure options and relaxation. Furthermore, computers have facilitated medical advancements, from precise diagnoses to accelerated research. However, there are downsides to this pervasive technology. Overreliance on computers can lead to reduced physical activity and potential health issues. Social isolation is another concern as face-to-face interactions decrease in favor of virtual connections."

#IMPORT DEPENDENCIES ################################################

##lexical features modules ===========================================
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk import download
from string import punctuation
from nltk.corpus import stopwords
#from nltk.corpus.stopwords import words

wn = WordNetLemmatizer() #specifying wn as the word net lemmatizer

#download nltk things
download('stopwords')
download('punkt')
download('wordnet')
download('averaged_perceptron_tagger')
stopwords_list = stopwords.words('english')

##syntactic features modules =========================================
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from collections import Counter
#import openai
# import pandas as pd
# import nltk

import numpy as np
from nltk.tree import Tree
import spacy

# Initialize spaCy English model
nlp_spacy = spacy.load('en_core_web_sm')

## Stylistic features models =============

from textblob import TextBlob

## passive sentence====
import textstat

# PREPROCESS TEXT #######################################

#remove punctuation, remove stopwords
def clean_text(text):
    #sentences = nltk.sent_tokenize(text) #create sentence tokens (not cleaned)
    text = "".join([word for word in text if word not in punctuation]) #remove punctuation
    tokens = word_tokenize(text) #tokenize
    text = [word for word in tokens if word not in stopwords_list] #remove stopwords
    return(text)

#create a function to lemmatize the text
def lemmatize_text(word_tokens):
    lem_text = [wn.lemmatize(word) for word in word_tokens]
    return(lem_text)

preproc_text = lemmatize_text(clean_text(test_text))


# FEATURE EXTRACTION ####################################################
from nltk import pos_tag

## Function to extract lexical features ======================================
def extract_lexical_features(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    total_word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / len(words)
    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    word_counts = Counter(words)
    TTR = len(word_counts) / len(words)
    stop_words = set(stopwords.words('english'))
    stop_word_count = sum(1 for word in words if word.lower() in stop_words)
    unique_word_count = sum(1 for _, count in word_counts.items() if count == 1)
    word_freq = word_counts
    bigram_freq = Counter(ngrams(words, 2))
    trigram_freq = Counter(ngrams(words, 3))
    rare_word_count = sum(1 for _, count in word_counts.items() if count == 1)

    return {
        'total_word_count': total_word_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'TTR': TTR,
        'stop_word_count': stop_word_count,
        'unique_word_count': unique_word_count,
        #'word_freq': word_freq,
        #'bigram_freq': bigram_freq,
        #'trigram_freq': trigram_freq,
        'rare_word_count': rare_word_count
    }

all_lexical_features = extract_lexical_features(test_text)

## Function to extract syntactic features =====================================
def extract_syntactic_features(text):
    # ... your extract_syntactic_features function implementation ...
    doc = nlp_spacy(text)

    # Calculate average sentence length
    sentence_lengths = [len(sent) for sent in doc.sents]
    avg_sentence_length = np.mean(sentence_lengths)

    # Calculate parse tree depth
    def calc_tree_depth(sent):
        root = [token for token in sent if token.head == token][0]
        return max([len(list(token.ancestors)) for token in sent])

    tree_depths = [calc_tree_depth(sent) for sent in doc.sents]
    avg_parse_tree_depth = np.mean(tree_depths)
    parse_tree_depth_variation = np.std(tree_depths)

    return {
        'avg_sentence_length': avg_sentence_length,
        'avg_parse_tree_depth': avg_parse_tree_depth,
        'parse_tree_depth_variation': parse_tree_depth_variation,
    }

all_syntactic_features = extract_syntactic_features(test_text)

##function to extract stylistic features ===============================================
def extract_stylistic_features(text):
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    pos_tagged_sentences = [pos_tag(sentence) for sentence in tokenized_sentences]
    
    num_adjectives = sum(sum(1 for word, pos in sentence if pos.startswith('JJ')) for sentence in pos_tagged_sentences)
    num_adverbs = sum(sum(1 for word, pos in sentence if pos.startswith('RB')) for sentence in pos_tagged_sentences)
    num_verbs = sum(sum(1 for word, pos in sentence if pos.startswith('VB')) for sentence in pos_tagged_sentences)
    num_nouns = sum(sum(1 for word, pos in sentence if pos.startswith('NN')) for sentence in pos_tagged_sentences)

    avg_adjectives_per_sentence = num_adjectives / num_sentences
    avg_adverbs_per_sentence = num_adverbs / num_sentences
    avg_verbs_per_sentence = num_verbs / num_sentences
    avg_nouns_per_sentence = num_nouns / num_sentences
    
    return {
        'avg_adjectives_per_sentence': avg_adjectives_per_sentence,
        'avg_adverbs_per_sentence': avg_adverbs_per_sentence,
        'avg_verbs_per_sentence': avg_verbs_per_sentence,
        'avg_nouns_per_sentence': avg_nouns_per_sentence,
    }

all_stylistic_features = extract_stylistic_features(test_text)

##function to calculate punctuation ========================
def count_punctuation(text):
    punctuation_count = sum(1 for char in text if char in punctuation)
    punct_length = sum(1 for char in text)
    punctuation_proportion = punctuation_count / punct_length
    return {"punctuation_proportion" :punctuation_proportion}

all_avg_punctuation = count_punctuation(test_text)

## Function to count passive sentences =========================
def count_passive_sentences(text):
    passive_sentences = 0
    doc = nlp_spacy(text)
    for token in doc:
        if token.dep_ == 'nsubjpass':
            passive_sentences += 1
    return {"passive_sentences" : passive_sentences}

passive_sentence_feature = count_passive_sentences(test_text)


## Function to calculate readability scores ==========================
def readability_scores(text):
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade_level = textstat.text_standard(text, float_output=True)
    smog_index = textstat.smog_index(text)
    return {
        "flesch_reading_ease" : flesch_reading_ease, 
        "flesch_kincaid_grade_level" : flesch_kincaid_grade_level, 
        "smog_index" : smog_index}

readability_feature = readability_scores(test_text)

## Function to calculate sentiment analysis scores ====================
def sentiment_analysis_scores(text):
    sentiment = TextBlob(text)
    return {
        "sentiment_polarity" : sentiment.polarity, 
        "sentiment_subjectivity" : sentiment.subjectivity}

sentiment_feature = sentiment_analysis_scores(test_text)

# COMBINE ALL FEATURES
features = {**all_lexical_features, **all_syntactic_features, **all_stylistic_features, **all_avg_punctuation, **passive_sentence_feature, **readability_feature, **sentiment_feature}
print(features)