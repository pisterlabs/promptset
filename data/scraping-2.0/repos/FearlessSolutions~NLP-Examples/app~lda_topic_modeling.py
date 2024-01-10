# python
import itertools
import os
import pprint
import random
import string

# external 
import matplotlib.pyplot as plt 
import pandas 
from wordcloud import WordCloud

# spacy 
import spacy
from spacy.lang.en import English, stop_words

# gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
 
def get_tokens(text):

    if type(text) != str:
        return " "

    text = text.replace("`", " ")
    my_tokens= nlp(text)

    # lemmatization (find the root words where necessary)
    my_tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in my_tokens ]

    punctuations = string.punctuation
    # Removing stop words and punctuations
    my_tokens = [ word for word in my_tokens if word not in stop_words.STOP_WORDS and word not in punctuations and "u0" not in word and word != "t" ]

    if "self" in my_tokens and "check" in my_tokens:
        my_tokens.remove("self")
        my_tokens.remove("check")
        my_tokens.append("selfcheck")
    if "data" in my_tokens:
        print("data")
        my_tokens.remove("data")

    if "science" in my_tokens:
        print("science")
        my_tokens.remove("science")

    if "machine" in my_tokens and "learning" in my_tokens:
        my_tokens.remove("machine")
        my_tokens.remove("learning")
        my_tokens.append("machine learning")

    return my_tokens

def generate_word_cloud(final_tokens):

    dummy_list = list()
    for key, value in final_tokens.items():
        dummy_list.extend([key] * value)

    random.shuffle(dummy_list)
    funk = ", ".join(dummy_list)
  
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(funk) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.savefig("/results/word_cloud.png")


def parse_lda_results(lda_model):
    '''
    The lda results come out in a strange string so this is a helper function to parse it
    Probably easier way to do this
    '''

    final = {}
    lda_results = lda_model.show_topics()[0][1]

    split_results = lda_results.split("+")

    for result in split_results:
        weight,entry  = result.split('*"')#[1].replace('"', " ").strip()
        entry = entry.replace('"', " ").strip()
        weight = int(float(weight)* 1000)

        if weight < 1:
            weight = 1

        final[entry] = weight

    return final

def create_bigram_trigram_models(data_words):
    '''
    data_words, list of lists, every element is an "article"
    '''

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=2, threshold=50) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return bigram_mod, trigram_mod

def tokenize(text):
    # Removing stop words and punctuations
    #my_tokens = [ word for word in my_tokens if word not in stop_words.STOP_WORDS and word not in punctuations and "u0" not in word and word != "t" ]

    tokens = gensim.utils.simple_preprocess(text, deacc=True)
 
    return tokens

def clean_text(text):
    no_stopwords = remove_stopwords(text)
    bigrams = make_bigrams(no_stopwords)

    lemmatized = lemmatization(bigrams)
    
    return lemmatized

def remove_stopwords(tokens):
    return [ word for word in tokens if word not in stop_words.STOP_WORDS and "u0" not in word and word != "t" and word != "andme" and word != "base" ]
    #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(tokens):
    return bigram_mod[tokens]

def make_trigrams(tokens):
    return trigram_mod[bigram_mod[tokens]]

def lemmatization(tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    # for sent in texts:
    doc = nlp(" ".join(tokens)) 
    
    return [token.lemma_ for token in doc if token.lemma_ != "datum"] 

def process_topics(all_data, articles):

    # Create Dictionary
    id2word = corpora.Dictionary(articles)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in articles]  

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    return lda_model

def process_survey():
    csv_path = os.environ["CSV_PATH"]
    topic_results_path = os.environ["TOPIC_RESULT_PATH"]
    column_key = os.environ["COLUMN_KEY"]

    df = pandas.read_csv(csv_path)

    # apply tokenize to every element
    tokenized_df = df[column_key].apply(tokenize) 

    all_data_corpus = tokenized_df.tolist() 

    # lazy, refactor later
    global bigram_mod 
    global trigram_mod
    bigram_mod, trigram_mod = create_bigram_trigram_models(all_data_corpus)

    # process ds examples 
    cleaned_text = tokenized_df.apply(clean_text)

    lda_model = process_topics("", cleaned_text.tolist())

    # write out LDA results
    with open(topic_results_path, "w") as fh:
        fh.write(pprint.pformat(lda_model.print_topics()))

    # work arounds to generate word cloud out of topics
    p_r = parse_lda_results(lda_model)

    # 
    generate_word_cloud(p_r)    

if __name__ == "__main__":
    process_survey()
  