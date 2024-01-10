# from twitterscraper import query_tweets
import pandas as pd
import re, pickle, os
import nltk
from nltk.util import ngrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords, wordnet 
from collections import Counter 
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.corpora import MmCorpus
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim

additional_stop_words=['hrtechconf','peopleanalytics','hrtech','hr','hrconfes',
                       'hrtechnology','voiceofhr','hrtechadvisor','gen','wait',
                       'next','see','hcm','booth','tech','la','vega','last',
                       'look','technology','work', 'announce','product','new',
                       'team','use','happen','time','take','make','everyone',
                       'anyone','week','day','year','let','go','come','word',
                       'employee','get','people','today','session','need',
                       'meet','help','talk','join','start','awesome','great',
                       'achieve','job','tonight','everyday','room','ready',
                       'one','company','say','well','data','share','love',
                       'want','like','good','business','sure','miss','demo',
                       'live','min','play','always','would','way','almost',
                       'thank','still','many','much','info','wow','play','full',
                       'org','create','leave','back','front','first','may',
                       'tomorrow','yesterday','find','stay','add','conference',
                       'top','stop','expo','hall','detail','row','award','hey',
                       'continue','put','part','whole','some','any','everywhere',
                       'convention','center','forget','congratulation','every',
                       'agenda','gift','card','available','behind','meeting',
                       'best','happen','unlockpotentialpic','half','none',
                       'human', 'resources','truly','win','possible','thanks',
                       'know','check','visit','fun','give','think','forward',
                       'twitter','com','pic','rt','via']

FIGURE_PATH = r'./figures/'
DATA_PATH = r'./data/'
MODEL_PATH = r'./models/'
WORDCLOUD_FILE = FIGURE_PATH + 'wordcloud.png'
WORD_COUNT_FILE = FIGURE_PATH + 'commond_words_freq.png'
TOPIC_VIS_FILE = FIGURE_PATH + 'lda.html'
ORIG_TWEET_FILE = DATA_PATH + 'all_tweets'
CLEANED_TWEET_FILE = DATA_PATH + 'tweets_cleaned_df'
CORPUS_FILE = MODEL_PATH + 'clean_tweets_corpus.mm'
DICT_FILE = MODEL_PATH + 'clean_tweets.dict'
LDA_MODEL_FILE = MODEL_PATH + 'tweets_lda.model'
LDA_TOPICS_FILE = MODEL_PATH + 'tweets_lda_topics.txt'
# ngrams or multi-word expressions
NUM_GRAMS = 2
# ----------------------
# LDA model parameters
# ----------------------
# Number of topics
NUM_TOPICS = 30
# Number of training passes
NUM_PASSES = 100 
# Document-Topic Density. The lower alpha is, the more likely that 
# a document may contain mixture of just a few of the topics. 
# Default is 1.0/NUM_TOPICS
ALPHA = 0.001
# Word-Topic Density. The lower eta is, the more likely that 
# a topic may contain a mixture of just a few of the words
# Default is 1.0/NUM_TOPICS
ETA = 'auto'
# ----------------------

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def text_cleanup(text):  
    '''
    Text pre-processing
        return tokenized list of cleaned words
    '''
    # Convert to lowercase
    text = str(text)
    text_clean = text.lower()
    # Remove non-alphabet
    text_clean = re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)',' ', text_clean).split()    
    # Remove short words (length < 3)
    text_clean = [w for w in text_clean if len(w)>2]
    # Lemmatize text with the appropriate POS tag
    lemmatizer = WordNetLemmatizer()
    text_clean = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text_clean]
    # Filter out stop words in English 
    stops = set(stopwords.words('english')).union(additional_stop_words)
    text_clean = [w for w in text_clean if w not in stops]
    
    return text_clean

def wordcloud(word_count_df):
    '''
    Create word cloud image
    '''
    # Convert DataFrame to Map so that word cloud can be generated from freq
    word_count_dict = {}
    for w, f in word_count_df.values:
        word_count_dict[w] = f
    # Generate word cloud 
    wordcloud = WordCloud(max_words=300, width=1400, height=900, 
                          random_state=12, contour_width=3, 
                          contour_color='firebrick')
    wordcloud.generate_from_frequencies(word_count_dict)
    plt.figure(figsize=(10,10), facecolor='k')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # Save the word cloud image
    wordcloud.to_file(WORDCLOUD_FILE) 
    print ('Word cloud saved\n')
    plt.close('all')
    
    return wordcloud
 
def read_data_from_pickle(infile):
    with open (infile, 'rb') as fp:
        return pickle.load(fp)
 
def save_data_to_pickle(outfile, all_tweets):
    with open(outfile, 'wb') as fp:
        pickle.dump(all_tweets, fp)
   
def save_print_to_file(outfile, msg):
    with open(outfile, 'w') as fp:
        print(msg, file=fp)

def preprocess_tweets(all_tweets_df):
    '''
    Preprocess tweets
    '''
    if os.path.isfile(CLEANED_TWEET_FILE):
        # Read cleaned tweets from saved file
        cleaned_tweets_df = read_data_from_pickle(CLEANED_TWEET_FILE)
        print('Loaded cleaned tweets from file\n')
    else:
        print('Start preprocessing tweets ...\n')
        # dataframe to add parsed tweets 
        cleaned_tweets_df = all_tweets_df.copy(deep=True)
        # parsing tweets 
        cleaned_tweets_df['token'] = [text_cleanup(x) for x in all_tweets_df['text']]     
        # Save cleaned tweets to file
        save_data_to_pickle(CLEANED_TWEET_FILE, cleaned_tweets_df)
        print ('Cleaned tweets saved\n')
    
    return cleaned_tweets_df 

def get_word_count(tweets_text, num_gram):
    '''
    Get common word counts
    '''
    n_grams = list(ngrams(tweets_text, num_gram))
    common_words = Counter(n_grams).most_common()
    word_count = pd.DataFrame(data = common_words, 
                              columns=['word','frequency']) 
    # Convert list to string
    word_count['word'] = word_count['word'].apply(' '.join)
    # Plot word count graph
    word_count.head(20).sort_values('frequency').plot.barh(
            x='word', y='frequency', title='Word Frequency',figsize=(19,10))
    plt.savefig(WORD_COUNT_FILE)
    print ('Word count saved\n')
    plt.close('all')
    
    return word_count

def word_grams(words, min=1, max=2):
    '''
    Build ngrams word list
    '''
    word_list = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            word_list.append(' '.join(str(i) for i in ngram))
    return word_list

def train_lda_model(token_tweets):
    print('Start LDA model training ...\n')    
    # Build dictionary
    tweets_dict = corpora.Dictionary(token_tweets)
    # Remove words that occur less than 10 documents, 
    # or more than 50% of the doc
    tweets_dict.filter_extremes(no_below=10, no_above=0.5)
    # Transform doc to a vectorized form by computing frequency of each word
    bow_corpus = [tweets_dict.doc2bow(doc) for doc in token_tweets]
    # Save corpus and dictionary to file
    MmCorpus.serialize(CORPUS_FILE, bow_corpus)
    tweets_dict.save(DICT_FILE)
    
    # Create tf-idf model and then apply transformation to the entire corpus
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    
    # Train LDA model
    lda_model = models.ldamodel.LdaModel(corpus=tfidf_corpus, 
                                         num_topics=NUM_TOPICS, 
                                         id2word=tweets_dict, 
                                         passes=NUM_PASSES, 
                                         alpha=ALPHA, 
                                         eta=ETA,
                                         random_state=49)
    # Save LDA model to file
    lda_model.save(LDA_MODEL_FILE)
    print ('LDA model saved\n')
    
    # Save all generated topics to a file
    msg = ''
    for idx, topic in lda_model.print_topics(-1):
        msg += 'Topic: {} \nWords: {}\n'.format(idx, topic)    
    save_print_to_file(LDA_TOPICS_FILE, msg)
    
    # Evaluate LDA model performance
    eval_lda (lda_model, tfidf_corpus, tweets_dict, token_tweets)    
    # Visualize topics
    vis_topics(lda_model, tfidf_corpus, tweets_dict)
        
    return lda_model

def eval_lda (lda_model, corpus, dict, token_text):
    # Compute Perplexity: a measure of how good the model is. lower the better.
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=token_text, 
                                         dictionary=dict, coherence='c_v')   
    print('\nCoherence: ', coherence_model_lda.get_coherence())

def vis_topics(lda_model, corpus, dict):
    '''
    Plot generated topics on an interactive graph
    '''
    lda_data =  pyLDAvis.gensim.prepare(lda_model, corpus, dict, mds='mmds')
    pyLDAvis.display(lda_data)
    pyLDAvis.save_html(lda_data, TOPIC_VIS_FILE)
    print ('Topic visual saved\n')

# Get all tweets
all_tweets_df = read_data_from_pickle(ORIG_TWEET_FILE)
all_tweets_df = pd.DataFrame(all_tweets_df)
# Preprocess tweets
cleaned_tweets_df = all_tweets_df.copy(deep=True)
cleaned_tweets_df['token'] = [text_cleanup(x) for x in all_tweets_df['full_text']]
# Save cleaned tweets and reload
save_data_to_pickle(CLEANED_TWEET_FILE, cleaned_tweets_df)
# cleaned_tweets_df = read_data_from_pickle(CLEANED_TWEET_FILE)

# Convert series to list for word count
tweets_text = [word for one_tweet in cleaned_tweets_df['token'] for word in one_tweet]
# Get common ngrams word count
word_count_df = get_word_count(tweets_text, num_gram=NUM_GRAMS)
# Generate word cloud
tweets_wordcloud = wordcloud(word_count_df)
# Generate ngram tokens

cleaned_tweets_df['ngram_token'] = [word_grams(x, NUM_GRAMS, NUM_GRAMS+1) for 
                 x in cleaned_tweets_df['token']]
save_data_to_pickle(MODEL_PATH + 'cleaned_tweets_ngrams', cleaned_tweets_df['ngram_token'])
# Train LDA model and visualize generated topics
lda_model = train_lda_model(cleaned_tweets_df['ngram_token'])
print('DONE!')
