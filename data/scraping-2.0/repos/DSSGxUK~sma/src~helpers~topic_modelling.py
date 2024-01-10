import logging

import gensim
import gensim.corpora as corpora
import pandas as pd
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
import os

os.environ["MALLET_HOME"] = "/files/mallet/mallet-2.0.8/"
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger('gensim').setLevel(logging.ERROR)

def create_dict(data, text_column):
    # Tokenize the docs
    tokenized_list = [simple_preprocess(doc) for doc in data[text_column]]

    # Create the Corpus and dictionary
    mydict = corpora.Dictionary()

    # The (0, 1) in line 1 means, the word with id=0 appears once in the 1st document.
    # Likewise, the (4, 4) in the second list item means the word with id 4 appears 4 times in the second document. And so on.
    mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]

    # Not human readable. Convert the ids to words.
    # Notice, the order of the words gets lost. Just the word and it?s frequency information is retained.
    word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]

    # Save the Dict and Corpus
    mydict.save('mydict.dict')  # save dict to disk
    corpora.MmCorpus.serialize('mycorpus.mm', mycorpus)  # save corpus to disk
    return mydict, mycorpus, tokenized_list


# Find the optimal number of topics for LDA.
# build many LDA models with different values of number of topics (k) 
# and pick the one that gives the highest coherence value.
# Choosing a k that marks the end of a rapid growth of topic coherence 
# usually offers meaningful and interpretable topics. 
# Picking an even higher value can sometimes provide more granular sub-topics.
# If the same keywords being repeated in multiple topics, it's probably a sign that the k is too large.
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    mallet_path = '/home/desktop1/files/mallet-2.0.8/mallet-2.0.8/bin/mallet'  
    
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=mycorpus, num_topics=num_topics, id2word=mydict)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    
    # Show graph
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    k = coherence_values.index(max(coherence_values))
    print(k)
    return model_list

def topic_scores(data: list, num_topics):
     # Tokenize the docs
    tokenized_list = [simple_preprocess(doc) for doc in data]

    # Create the Corpus and dictionary
    mydict = corpora.Dictionary()

    # The (0, 1) in line 1 means, the word with id=0 appears once in the 1st document.
    # Likewise, the (4, 4) in the second list item means the word with id 4 appears 4 times in the second document. And so on.
    mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
    mallet_path = '/files/mallet-2.0.8/mallet-2.0.8/bin/mallet'  
    model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=mycorpus, num_topics=num_topics, id2word=mydict)

    #coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    #coherence_values.append(coherencemodel.get_coherence())
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get topic in each document
    for i, row in enumerate(model[mycorpus]):
        # Get the topics, Perc Contribution and for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            sent_topics_df = sent_topics_df.append(pd.Series([i, int(topic_num), prop_topic]), ignore_index=True)
    sent_topics_df.columns = ['row_number','Topic', 'Topic_Contribution']
    sent_topics_df = sent_topics_df.pivot(index="row_number", columns="Topic", values="Topic_Contribution").reset_index()
    return sent_topics_df
