import json
import os
from collections import defaultdict

import numpy as np
import gensim
from gensim import corpora
from gensim.models import CoherenceModel

from lib2vec.corpus_iterators import TopicModellingIterator
from lib2vec.corpus_structure import Corpus
from lib2vec.vectorization_utils import Vectorization


class TopicModeller:
    @staticmethod
    def compute_coherence_values(dictionary, corpus, texts, limit, start, step, id2word, mallet_path: str = None):
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
        number_topics = []
        for num_topics in range(start, limit, step):
            if mallet_path:
                model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics,
                                                         id2word=id2word)
            else:
                model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                        id2word=id2word,
                                                        num_topics=num_topics,
                                                        random_state=100,
                                                        update_every=1,
                                                        chunksize=100,
                                                        passes=10,
                                                        alpha='auto',
                                                        per_word_topics=True)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
            number_topics.append(number_topics)

        # plt.plot(range(start, limit, step), coherence_values)
        # plt.xlabel("Num Topics")
        # plt.ylabel("Coherence score")
        # plt.legend(("coherence_values"), loc='best')
        # plt.show()
        best_model_index = coherence_values.index(max(coherence_values))
        best_topics = number_topics[best_model_index]
        return model_list[best_model_index], max(coherence_values), best_topics

    @staticmethod
    def get_topic_words_for_docs(lda_model, corpus, id2doc_id):
        word_dist = defaultdict(list)
        for d_id, doc in enumerate(corpus):
            # topic_dist = [topic_dist for topic_dist in lda_model[doc][0]]
            topic_dist = sorted(lda_model[doc][0], key=lambda x: (x[1]), reverse=True)
            print(topic_dist)
            topics = [tup[0] for tup in topic_dist[:5] if tup[1] > 0.05]
            print(topics)
            for topic_num in topics:
                words_in_topics = lda_model.show_topic(topic_num, 100)
                words_in_topics = [word for (word, perc) in words_in_topics]
                word_dist[id2doc_id[d_id]].extend(words_in_topics)

        # for i, row in enumerate(lda_model[corpus][doc_id][0]):
        #     row = sorted(row, key=lambda x: (x[1]), reverse=True)
        #     # Get the Dominant topic, Perc Contribution and Keywords for each document
        #     for j, (topic_num, prop_topic) in enumerate(row):
        #         if j == 0:  # => dominant topic
        #             wp = lda_model.show_topic(topic_num)
        #             topic_keywords = ", ".join([word for word, prop in wp])
        #             sent_topics_df = sent_topics_df.append(
        #                 pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
        #         else:
        #             break
        # sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        # contents = pd.Series(texts)
        # sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return word_dist

    @staticmethod
    def train_lda(corpus: Corpus):
        # def make_bigrams(texts):
        #     return [bigram_mod[doc] for doc in texts]

        def make_trigrams(texts):
            return [trigram_mod[bigram_mod[doc]] for doc in texts]
        # c.filter("ne")
        # c.filter("V")
        corpus = corpus.filter_on_copy("stopwords")
        corpus = corpus.filter_on_copy("punctuation")
        # data_words = [document.get_flat_document_tokens(lemma=True, lower=True)
        #               for doc_id, document in c.documents.items()]
        data_words = corpus.get_flat_document_tokens(lemma=True, lower=True)
        id2doc_id = {i: doc_id for i, doc_id in enumerate(corpus.documents.keys())}

        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        trigram = gensim.models.Phrases(bigram[data_words], threshold=150)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        data_lemmatized = make_trigrams(data_words)

        id2word = corpora.Dictionary(data_lemmatized)
        corpus = [id2word.doc2bow(text) for text in data_lemmatized]

        # limit = 40
        # start = 2
        # step = 6
        # lda_model, coherence, num_topics = compute_coherence_values(dictionary=id2word,
        #                                                             corpus=corpus,
        #                                                             texts=data_lemmatized,
        #                                                             start=start,
        #                                                             limit=limit,
        #                                                             step=step,
        #                                                             id2word)
        # print(coherence, num_topics)

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=15,
                                                    random_state=100,
                                                    update_every=1,
                                                    iterations=100,
                                                    chunksize=100,
                                                    passes=50,
                                                    alpha='auto',
                                                    minimum_probability=0.0,
                                                    per_word_topics=True)

        # os.environ.update({'MALLET_HOME': r'C:/mallet_new/mallet-2.0.8'})
        # mallet_path = "bin\\mallet"
        # print(mallet_path)
        # lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=15, id2word=id2word,
        #                                              alpha='auto', random_seed=42)

        # print(lda_model.print_topics())

        # Compute Coherence Score
        # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word,
        #                                      coherence='c_v')
        # coherence_lda = coherence_model_lda.get_coherence()
        # print('\nCoherence Score: ', coherence_lda)

        content_aspect_dict = TopicModeller.get_topic_words_for_docs(lda_model, corpus, id2doc_id)
        content_aspect_list = [texts for doc_id, texts in content_aspect_dict.items()]
        # print(content_aspect_list)
        # print(content_aspect_dict)
        return content_aspect_dict, content_aspect_list

    @staticmethod
    def train_lda_mem_eff(corpus: Corpus):
        # def make_bigrams(texts):
        #     return [bigram_mod[doc] for doc in texts]
        #
        # def make_trigrams(texts):
        #     return [trigram_mod[bigram_mod[doc]] for doc in texts]

        # c.filter("ne")
        # c.filter("V")

        id2doc_id = {i: doc_id for i, doc_id in enumerate(corpus.documents.keys())}
        lemma = True
        lower = False
        # print('vocab_start')
        # vocab = set(TokenIterator(corpus, lemma=lemma, lower=lower))
        # print(len(list(vocab)))
        # print('vocab gen end')
        vocab = [[token for token in document.get_vocab(from_disk=True, lemma=lemma, lower=lower, lda_mode=True)
                  if token != 'del']
                 for doc_id, document in corpus.documents.items()]
        # print(len(list(vocab)))
        # print('vocab end')
        # vocab = corpus.get_corpus_vocab(lemma=lemma, lower=lower,
        #                                 lda_mode=True)

        id2word_dict = corpora.Dictionary(list(vocab))

        corpus = TopicModellingIterator(corpus, id2word_dict, lemma=lemma, lower=lower)
        # print(corpus)
        # data_words = [document.get_flat_document_tokens(lemma=True, lower=True)
        #               for doc_id, document in c.documents.items()]
        # data_words = corpus.get_flat_document_tokens(lemma=True, lower=True)
        # bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        # bigram_mod = gensim.models.phrases.Phraser(bigram)
        #
        # trigram = gensim.models.Phrases(bigram[data_words], threshold=150)
        # trigram_mod = gensim.models.phrases.Phraser(trigram)
        #
        # data_lemmatized = make_trigrams(data_words)
        #
        # id2word_dict = corpora.Dictionary(data_lemmatized)
        # corpus = [id2word_dict.doc2bow(text) for text in data_lemmatized]
        # limit = 40
        # start = 2
        # step = 6
        # lda_model, coherence, num_topics = compute_coherence_values(dictionary=id2word_dict,
        #                                                             corpus=corpus,
        #                                                             texts=data_lemmatized,
        #                                                             start=start,
        #                                                             limit=limit,
        #                                                             step=step,
        #                                                             id2word_dict)
        # print(coherence, num_topics)

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word_dict,
                                                    num_topics=15,
                                                    random_state=100,
                                                    update_every=1,
                                                    iterations=50,
                                                    chunksize=100,
                                                    passes=50,
                                                    alpha='auto',
                                                    minimum_probability=0.0,
                                                    per_word_topics=True)
        # print('calc')
        # os.environ.update({'MALLET_HOME': r'C:/mallet_new/mallet-2.0.8'})
        # mallet_path = "bin\\mallet"
        # print(mallet_path)
        # lda_model = gensim.models.wrappers.LdaMallet(mallet_path,
        #                                              corpus=corpus, num_topics=15, id2word_dict=id2word_dict,
        #                                              alpha='auto', random_seed=42)

        # print(lda_model.print_topics())

        # Compute Coherence Score
        # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word_dict,
        #                                      coherence='c_v')
        # coherence_lda = coherence_model_lda.get_coherence()
        # print('\nCoherence Score: ', coherence_lda)

        content_aspect_dict = TopicModeller.get_topic_words_for_docs(lda_model, corpus, id2doc_id)
        content_aspect_list = [texts for doc_id, texts in content_aspect_dict.items()]
        # print(content_aspect_list)
        # print(content_aspect_dict)
        return content_aspect_dict, content_aspect_list, lda_model, list(corpus), corpus.doc_ids

    @staticmethod
    def topic_modelling(corpus: Corpus):
        topic_dict_path = os.path.join(corpus.corpus_path, "topic_ids.json")
        if not os.path.isfile(topic_dict_path):
            print("train topic model")
            topic_dict, _, _, _, _ = TopicModeller.train_lda_mem_eff(corpus)

            with open(topic_dict_path, 'w', encoding='utf-8') as fp:
                json.dump(topic_dict, fp, indent=1)
        else:
            with open(topic_dict_path) as json_file:
                topic_dict = json.load(json_file)

        return topic_dict

    @staticmethod
    def get_topic_distribution(corpus: Corpus, dataset: str, overwrite: bool = False):
        if overwrite or not os.path.isfile(f'D:/models/topic_vectors/{dataset}.kv'):
            _, _, topic_model, lda_corpus, doc_ids = TopicModeller.train_lda_mem_eff(corpus)
            topic_vectors = {}
            # print(len(lda_corpus))
            # print(doc_ids)
            for i, doc_id in enumerate(doc_ids):
                doc = lda_corpus[i]
                topic_vectors[doc_id] = np.array([score for (topic, score) in topic_model[doc][0]])

            # print(topic_vectors)
            Vectorization.my_save_doc2vec_format(fname=f'D:/models/topic_vectors/{dataset}.kv',
                                                 doctag_vec=topic_vectors)

        topic_vecs, _ = Vectorization.my_load_doc2vec_format(f'D:/models/topic_vectors/{dataset}.kv')

        # print(topic_vecs.docvecs.doctags)
        # for doctag in topic_vecs.docvecs.doctags:
        #     print(doctag, topic_vecs.docvecs.most_similar(doctag, topn=None))
        # print(topic_model[lda_corpus[0]])
        # for document in topic_model:
        #     doc_id = ...
        #     gensim_doc_id = ...
        #     topic_vectors[doc_id] = topic_model[lda_corpus[gensim_doc_id]]
        return topic_vecs


if __name__ == "__main__":
    data_set_name = "classic_gutenberg"
    c = Corpus.load_corpus_from_dir_format(os.path.join(f"corpora/{data_set_name}"))
    # d = TopicModeller.train_lda(c)
    TopicModeller.get_topic_distribution(c, data_set_name, overwrite=True)
