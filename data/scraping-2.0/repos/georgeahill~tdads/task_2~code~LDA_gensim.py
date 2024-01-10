# Implementation of topic detection in tweets using Latent Dirichlet Allocation
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from preproc import *

# Grid search parameters
TOPIC_RANGE = range(3, 7)
ALPHA_RANGE = [0.5]
BETA_RANGE = [0.5]

# Hyperparameters (from grid search)
NUM_TOPICS = 8
ALPHA = 0.5
BETA = 0.5

FILTERED_WORDS = ["airline", "otherairline", "user", "-pron-"]
OUT_DIR = "../code_output/airline_topics/"

def pd_read(filename = "tweets.csv", lower = True):
    """ Read tweets from filename
    
    Parameters:
        filename (str)
        lower (bool): optional lowercase

    Returns:
        pandas.DataFrame()
    """
    tweets = pd.read_csv(filename)
    tweets.drop_duplicates(subset='text', inplace=True)
    if lower:
        tweets.text = tweets.text.str.lower()
    return tweets

def preprocess(tweets):
    # Get only negative ones (for this task)
    newTweets = tweets.copy()

    newTweets = remove_airline_tags(newTweets)
    newTweets.text = remove_links(newTweets.text)
    newTweets.text = lt_gt_conversion(ampersand_conversion(arrow_conversion(newTweets.text)))
    newTweets.text = with_without_conversion(newTweets.text)
    newTweets.text = hashtag_to_words(newTweets.text)
    newTweets = translate_all_emoji(newTweets)
    newTweets.text = remove_contractions(newTweets.text)
    newTweets.text = remove_punctuation(newTweets.text)
    newTweets.text = lemmatize_texts(newTweets.text)
    newTweets.text = remove_stopwords(newTweets.text)
    newTweets.text = newTweets.text.str.lower()
    texts = newTweets["text"].values
    
    # Tokenize and remove short words or filtered words
    tokenized_texts = []
    for text in texts:
        split_text = text.split()
        split_text = [word for word in split_text if len(word) > 2 and word not in FILTERED_WORDS]
        tokenized_texts.append(split_text)

    # Create a dictionary for each word, and a bag of words
    text_dictionary = Dictionary(tokenized_texts)

    # Remove words that appear in over 50%, or less than 0.5%, and keep the top 66% of the vocabulary
    text_dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=len(text_dictionary)//2)
    text_corpus = [text_dictionary.doc2bow(text) for text in tokenized_texts]
    return (text_dictionary, text_corpus)

# Assumes that unprocessed data is provided
# Data is a pandas dataframe in the form of the tweets.csv file
def fit(tweets, alpha_value=ALPHA, beta_value=BETA, num_topics_value=NUM_TOPICS):
    dictionary, corpus = preprocess(tweets)

    # Replace with gensim.models.ldamodel.LdaModel if this causes issues
    lda = LdaMulticore(corpus, num_topics=num_topics_value, id2word=dictionary, passes=10, alpha=alpha_value, eta=beta_value)
    return [lda, dictionary, corpus]

def predict(tweets, fit_return_list):
    lda = fit_return_list[0]
    dictionary = fit_return_list[1]
    corpus = fit_return_list[2]
    predictions = []
    # LdaModel prediction returns an array of tuples in the form (TOPIC_NUM, PROBABILITY)
    for probabilities in lda[corpus]:
        predictions.append(max(probabilities, key=lambda x: x[1])[0])
    return predictions

def getModelCoherence(dictionary, corpus, topic, a, b):
    # Remaking the provided texts fixes an error
    reconstructedTexts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]
    
    # Replace with gensim.models.ldamodel.LdaModel if this causes issues
    lda = LdaMulticore(corpus, num_topics=topic, id2word=dictionary, passes=10, alpha=a, eta=b)
    coherence_model = CoherenceModel(model=lda, texts=reconstructedTexts, dictionary=dictionary, coherence="c_v")
    coherence = coherence_model.get_coherence()
    return coherence
        
def gridSearch(tweets, verbose=0):
    dictionary, corpus = preprocess(tweets)

    results = {"Topics" : [], "Alpha" : [], "Beta" : [], "Coherence" : []}

    NUM_PARAMETERS = str(len(TOPIC_RANGE) * len(ALPHA_RANGE) * len(BETA_RANGE))
    if verbose == 1:
        print("Total number of parameters to test: " + NUM_PARAMETERS)

    count = 1
    for topic in TOPIC_RANGE:
        for alpha in ALPHA_RANGE:
            for beta in BETA_RANGE:
                if verbose == 1:
                    print(str(count) + "/" + NUM_PARAMETERS + ": alpha=" + str(alpha) + " beta=" + str(beta) + " num_topics=" + str(topic))
                coherence = getModelCoherence(dictionary, corpus, topic, alpha, beta)
                results["Topics"].append(topic)
                results["Alpha"].append(alpha)
                results["Beta"].append(beta)
                results["Coherence"].append(coherence)
                count += 1
    if verbose == 1:
        print("Finished grid search.")

    results_df = pd.DataFrame(data=results)
    bestValues = results_df["Coherence"].argmax()
    return results_df.at[bestValues, "Alpha"], results_df.at[bestValues, "Beta"], results_df.at[bestValues, "Topics"]
    
    
    

if __name__ == "__main__":
    tweets = pd_read("tweets.csv")
    tweets = tweets[tweets.airline_sentiment == "negative"]
    airline_dict = dict(tuple(tweets.groupby("airline")))
    by_airline = [airline_dict[x] for x in airline_dict]

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    for airline in by_airline:
        print("Finding optimal number of topics for " + airline["airline"].iloc[0])
        alpha, beta, topics = gridSearch(airline, verbose=1)
        print("Fitting")
        return_list = fit(airline, alpha, beta, topics)
        print("Testing on unseen tweets.")
        predictions = predict(airline, return_list)
        print("Saving to " + OUT_DIR + ".csv")
        airline["predictions"] = predictions
        airline.to_csv(OUT_DIR + airline["airline"].iloc[0] + ".csv", index=False)
    



    
