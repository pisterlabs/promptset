from gensim.models import CoherenceModel
import pickle
import numpy as np


def getTopics(model, top_clusters, n_words):

    # create empty list to contain topics
    topics = []

    # iterate over top n clusters
    for cluster in top_clusters:
        # create sorted dictionary of word distributions
        sorted_dict = sorted(model.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[
                      :n_words]

        # create empty list to contain words
        topic = []

        # iterate over top n words in topic
        for k, v in sorted_dict:
            # append words to topic list
            topic.append(k)

        # append topics to topics list
        topics.append(topic)

    return topics

if __name__ == "__main__":

    with open('./models/gsdmm_models/model.pkl', 'rb') as f:
        gsdmm = pickle.load(f)

    with open('./models/gsdmm_models/model_corpus.pkl', 'rb') as f:
        bow_corpus = pickle.load(f)

    with open('./models/gsdmm_models/model_dict.pkl', 'rb') as f:
        dictionary = pickle.load(f)

    with open('./models/gsdmm_models/model_texts.pkl', 'rb') as f:
        docs = pickle.load(f)

    doc_count = np.array(gsdmm.cluster_doc_count)

    # Topics sorted by the number of document they are allocated to
    top_index = doc_count.argsort()[-15:][::-1]

    # get topics to feed to coherence model
    topics = getTopics(gsdmm, top_index, 20)

    # evaluate model using Topic Coherence score
    cm_gsdmm = CoherenceModel(topics=topics,
                            dictionary=dictionary,
                            corpus=bow_corpus,
                            texts=docs,
                            coherence='c_v')

    # get coherence value
    coherence_gsdmm = cm_gsdmm.get_coherence()

    print(coherence_gsdmm)