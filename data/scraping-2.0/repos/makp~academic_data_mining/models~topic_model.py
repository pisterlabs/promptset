"""Functions for topic modeling."""

from gensim.models.coherencemodel import CoherenceModel


def get_lda_topics_without_prob(lda_model):
    """Return the `num_topics` topics of an LDA model without probabilities."""
    num_topics = lda_model.num_topics
    lda_topics = []
    for topic_id in range(num_topics):
        topic = lda_model.show_topic(topic_id)
        topic_words = [word for (word, _) in topic]
        lda_topics.append(topic_words)

    return lda_topics


def get_nmf_topics_without_prob(nmf_model):
    """Return the `num_topics` topics of an NMF model without probabilities."""
    num_topics = nmf_model.num_topics
    nmf_topics = []
    for topic_id in range(num_topics):
        topic = nmf_model.show_topic(topic_id)
        topic_words = [word for word, _ in topic]
        nmf_topics.append(topic_words)

    return nmf_topics


def compute_coherence(topics,
                      tokenized_docs,
                      dictionary):
    """
    Compute the coherence of a topic model using c_v and u_mass.

    Sliding window methods (e.g., 'c_v') do not require the corpus
    but they require tokenized texts. 'u_mass' requires a corpus but
    not tokenized texts. However, Gensim uses the dictionary and the
    tokenized docs to generate the corpus if it is not provided.
    Moreover, not having 'corpus' as an argument avoids the risk of
    calculating the coherence score with TF-IDF vectors (TF-IDF
    vectors might distort the co-occurrence information that u_mass
    relies upon).

    Roughly, while c_v favors topics that are distinct but maybe not
    very specific, u_mass favors topics that are tightly focused but
    possibly overlapping.
    """
    cm_cv = CoherenceModel(topics=topics,
                           texts=tokenized_docs,
                           dictionary=dictionary,
                           coherence='c_v')

    cm_umass = CoherenceModel(topics=topics,
                              texts=tokenized_docs,
                              dictionary=dictionary,
                              coherence='u_mass')

    return cm_cv.get_coherence(), cm_umass.get_coherence()


def list_topics_for_bow_sorted(lda_model, bow):
    """
    Return the list of topics for a bow vector.

    The list is sorted by the probability of each topic.
    """
    topic_dist = lda_model.get_document_topics(bow,
                                               minimum_probability=None)
    return sorted(topic_dist, key=lambda x: x[1], reverse=True)


def get_top_n_docs_for_topic(series_bow,
                             lda_model,
                             topic_id,
                             num_docs=1):
    """Get the indexes of the top `num_docs` documents for a given topic."""
    lst = []
    for idx, bow in series_bow.items():
        topic_dist = dict(lda_model.get_document_topics(bow))
        prob = topic_dist.get(topic_id, 0)
        lst.append((idx, prob))
    lst.sort(key=lambda x: x[1], reverse=True)
    return lst[:num_docs]


# def compute_perplexity(lda_model, test_corpus):
#     """
#     Compute and return the perplexity of a LDA model for a holdout
#     corpus.
#     Perplexity measures how well a probability model predicts a
#     sample. The lower the score, the better.
#     """
#     num_docs = len(test_corpus)
#     unnormalized_log_perplexity = 0
#     for doc_bow in test_corpus:
#         doc_topics = lda_model.get_document_topics(doc_bow,
#                                                    minimum_probability=0)
#         doc_probs = [prob for (_, prob) in doc_topics]
#         doc_perplexity = -np.dot(np.log(doc_probs), doc_probs)
#         unnormalized_log_perplexity += doc_perplexity
#     normalized_log_perplexity = unnormalized_log_perplexity / num_docs
#     perplexity = np.exp(normalized_log_perplexity)

#     return perplexity
