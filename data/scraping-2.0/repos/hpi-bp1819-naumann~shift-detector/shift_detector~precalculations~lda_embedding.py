import pandas as pd
import numpy as np
from numbers import Number
from sklearn.decomposition import LatentDirichletAllocation as LDA_skl
from sklearn.feature_extraction.text import *
from gensim.sklearn_api import LdaTransformer
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import warnings
from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.precalculations.count_vectorizer import CountVectorizer
from shift_detector.precalculations.lda_gensim_tokenizer import LdaGensimTokenizer
from shift_detector.utils.column_management import ColumnType


class LdaEmbedding(Precalculation):

    def __init__(self, columns, n_topics=20, n_iter=10, random_state=0, lib='sklearn', trained_model=None,
                 start=2, stop=21, step=1, stop_words='english', max_features=None):
        self.model = None
        self.trained_model = None
        self.lib = None
        self.columns = None
        self.stop_words = stop_words
        self.max_features = max_features

        if columns:
            if isinstance(columns, list) and all(isinstance(col, str) for col in columns):
                self.columns = columns
            else:
                raise TypeError("Columns has to be list of strings . Column {} is of type {}"
                                .format(columns, type(columns)))
        else:
            raise ValueError("You have to specify which columns you want to vectorize")

        if trained_model:
            warnings.warn("Trained models are not trained again. Please make sure to only input the column(s) "
                          "that the model was trained on", UserWarning)
            self.trained_model = trained_model
            self.random_state = self.trained_model.random_state
            if isinstance(self.trained_model, type(LDA_skl())):
                self.n_topics = self.trained_model.n_components
                self.n_iter = self.trained_model.max_iter
            else:
                self.n_topics = self.trained_model.num_topics
                self.n_iter = self.trained_model.iterations
        else:
            if n_topics == 'auto':
                self.n_topics = n_topics
                params = [start, stop, step]
                for number in params:
                    try:
                        val = int(number)
                        if val < 2:
                            raise ValueError("Number of topic has to be a positive. Received: {}".format(number))
                        break
                    except TypeError:
                        raise TypeError("That's not an int! Received: {}".format(type(number)))
                if stop < start:
                    raise ValueError("Stop value has to be higher than the start value. Received: {}".format(n_topics))
                self.start = start
                self.stop = stop
                self.step = step
            else:
                if not isinstance(n_topics, int):
                    raise TypeError("Number of topic has to be an integer. Received: {}".format(type(n_topics)))
                if n_topics < 2:
                    raise ValueError("Number of topics has to be at least 2. Received: {}".format(n_topics))
                self.n_topics = n_topics

            if not isinstance(n_iter, int):
                raise TypeError("Random_state has to be a integer. Received: {}".format(type(n_iter)))
            if n_iter < 1:
                raise ValueError("Random_state has to be at least 1. Received: {}".format(n_iter))
            self.n_iter = n_iter

            if not isinstance(random_state, int):
                raise TypeError("Random_state has to be a integer. Received: {}".format(type(random_state)))
            if random_state < 0:
                raise ValueError("Random_state has to be positive or zero. Received: {}".format(random_state))
            self.random_state = random_state

            if not isinstance(lib, str):
                raise TypeError("Lib has to be a string. Received: {}".format(type(lib)))
            if lib == 'sklearn':
                self.model = \
                    LDA_skl(n_components=self.n_topics, max_iter=self.n_iter, random_state=self.random_state)
            elif lib == 'gensim':
                self.model = \
                    LdaTransformer(num_topics=self.n_topics, iterations=self.n_iter, random_state=self.random_state)
            else:
                raise ValueError("The supported libraries are sklearn and gensim. Received: {}".format(lib))
        self.lib = lib

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            model_attributes = sorted([(k, v) for k, v in self.model.__dict__.items()
                                       if isinstance(v, Number) or isinstance(v, str) or isinstance(v, list)])
            other_model_attributes = sorted([(k, v) for k, v in other.model.__dict__.items()
                                             if isinstance(v, Number) or isinstance(v, str) or isinstance(v, list)])
            return isinstance(other.model, self.model.__class__) \
                and model_attributes == other_model_attributes and self.columns == other.columns \
                and self.stop_words == other.stop_words and self.max_features == other.max_features
        return False

    def __hash__(self):
        if self.trained_model:
            trained_hash_list = [self.__class__, self.trained_model.__class__]
            for item in self.trained_model.__dict__.items():
                if not item[0] == 'components_' and not item[0] == 'exp_dirichlet_component_':
                    # dirty fix I know, ndarrays are not hashable
                    trained_hash_list.extend(item)
            return hash(tuple(trained_hash_list))
        elif self.columns:
            hash_list = [self.__class__, self.model.__class__, self.n_topics,
                         self.n_iter, self.random_state, self.max_features]
            hash_list.extend(self.columns)
            hash_list.extend(self.stop_words)
            return hash(tuple(hash_list))
        else:
            return hash(tuple([self.__class__, self.model.__class__, self.n_topics,
                               self.n_iter, self.random_state]))

    @staticmethod
    def topic_probabilities_to_topics(lda_model, dtm):
        # Always takes the topic with the highest probability as the dominant topic
        return [arr.argmax()+1 for arr in lda_model.transform(dtm)]

    @staticmethod
    def get_topic_word_distribution_gensim(lda_model, n_topics, n_top_words):
        topic_words = lda_model.gensim_model.show_topics(num_topics=n_topics,
                                                         num_words=n_top_words,
                                                         formatted=False)
        return topic_words

    @staticmethod
    def get_topic_word_distribution_sklearn(lda_model, vocab, n_top_words):
        # copied implementation from gensim show_topics
        topic_words = []
        for topic_n, comp in enumerate(lda_model.components_):
            topic_ = comp
            topic_ = topic_ / topic_.sum()
            most_extreme = np.argpartition(-topic_, n_top_words)[:n_top_words]
            word_idx = most_extreme.take(np.argsort(-topic_.take(most_extreme)))
            topic_ = [(vocab[id], topic_[id]) for id in word_idx]
            topic_words.append((topic_n, topic_))
        return topic_words

    def get_number_of_topics_with_best_coherence_score(self, col, tokenized_merged, all_corpora, all_dicts):
        coherence_scores = {}
        for n in range(self.start, self.stop, self.step):
            model = LdaModel(all_corpora[col], n, all_dicts[col], random_state=0)
            cm = CoherenceModel(model=model, texts=tokenized_merged[col], coherence='c_v')
            coherence = cm.get_coherence()
            coherence_scores[n] = coherence
        return max(coherence_scores, key=lambda k: coherence_scores[k])

    def process(self, store):
        if isinstance(self.columns, str):
            if self.columns in store.column_names(ColumnType.text):
                col_names = self.columns
        else:
            for col in self.columns:
                if col not in store.column_names(ColumnType.text):
                    raise ValueError("Given column is not contained in detected text columns of the datasets: {}"
                                     .format(col))
            col_names = self.columns

        topic_labels = ['topics ' + col for col in col_names]

        transformed1 = pd.DataFrame()
        transformed2 = pd.DataFrame()
        topic_words_all_columns = {}
        all_models = {}

        if self.lib == 'gensim':
            tokenized1, tokenized2 = store[LdaGensimTokenizer(stop_words=self.stop_words, columns=self.columns)]
            tokenized_merged = pd.concat([tokenized1, tokenized2], ignore_index=True)

            all_corpora = {}
            all_dicts = {}

            for i, col in enumerate(col_names):
                all_dicts[col] = Dictionary(tokenized_merged[col])
                gensim_dict1 = Dictionary(tokenized1[col])
                gensim_dict2 = Dictionary(tokenized2[col])

                all_corpora[col] = [all_dicts[col].doc2bow(line) for line in tokenized_merged[col]]
                corpus1 = [gensim_dict1.doc2bow(line) for line in tokenized1[col]]
                corpus2 = [gensim_dict2.doc2bow(line) for line in tokenized2[col]]

                if not self.trained_model:
                    if self.n_topics == 'auto':
                        n_topics = self.get_number_of_topics_with_best_coherence_score(col, tokenized_merged,
                                                                                       all_corpora, all_dicts)
                        self.model.num_topics = n_topics
                    else:
                        n_topics = self.n_topics

                    model = self.model
                    model.id2word = all_dicts[col]
                    model = model.fit(all_corpora[col])
                    all_models[col] = model.gensim_model
                else:
                    model = self.trained_model

                topic_words_all_columns[col] = self.get_topic_word_distribution_gensim(model, n_topics, 200)

                transformed1[topic_labels[i]] = self.topic_probabilities_to_topics(model, corpus1)
                transformed2[topic_labels[i]] = self.topic_probabilities_to_topics(model, corpus2)

            return transformed1, transformed2, topic_words_all_columns, all_models, all_corpora, all_dicts

        else:
            vectorized1, vectorized2, feature_names, all_vecs = store[CountVectorizer(stop_words=self.stop_words,
                                                                                      max_features=self.max_features,
                                                                                      columns=self.columns)]
            all_dtms = dict(vectorized1, **vectorized2)
            if self.n_topics == 'auto':
                tokenized1, tokenized2 = store[LdaGensimTokenizer(stop_words=self.stop_words, columns=self.columns)]
                tokenized_merged = pd.concat([tokenized1, tokenized2], ignore_index=True)

                all_corpora = {}
                all_dicts = {}

            for i, col in enumerate(col_names):
                if not self.trained_model:
                    if self.n_topics == 'auto':
                        all_dicts[col] = Dictionary(tokenized_merged[col])
                        all_corpora[col] = [all_dicts[col].doc2bow(line) for line in tokenized_merged[col]]
                        n_topics = self.get_number_of_topics_with_best_coherence_score(col, tokenized_merged,
                                                                                       all_corpora, all_dicts)
                        self.model.n_components = n_topics
                    model = self.model
                    model = model.fit(all_dtms[col])
                    all_models[col] = model
                else:
                    model = self.trained_model

                topic_words_all_columns[col] = self.get_topic_word_distribution_sklearn(model, feature_names[col], 200)
                transformed1[topic_labels[i]] = \
                    self.topic_probabilities_to_topics(model, vectorized1[col])
                transformed2[topic_labels[i]] = \
                    self.topic_probabilities_to_topics(model, vectorized2[col])

            return transformed1, transformed2, topic_words_all_columns, all_models, all_dtms, all_vecs
