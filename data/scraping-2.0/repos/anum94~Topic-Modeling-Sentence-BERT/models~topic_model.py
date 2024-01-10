# Importing the required python modules
from models.embeddings_model import get_embeddings
from models.autoencoder_model import Autoencoder
from utilities.utils import generate_n_gram, clean_topics_from_clusters
from utilities.visualizations import get_wordcloud, create_hist, vis_3d, vis_2d
from datetime import datetime
import gensim
from gensim import corpora, models
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import Counter
import torch

import os
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize

nltk.download("wordnet")


class Topic_Model:
    def __init__(
        self,
        base="LDA",
        k=3,
        use_AE=True,
        gamma=1,
        embedding=None,
        saved_embedding_dir=None,
        verbose=False,
        embedding_batch_size=None,
        saved_attention_dir=None,
        output_attention=False,
    ):
        """

        :param base (str): Probabilistic algorithm to be used. possible values: 'LDA'
        :param k (int): expected number of clusters in the data
        :param AE (bool): whether or not to use AutoEncoder
        :param embedding (str): which embedding to use. possible values 'bert', 'albert', 'roberta', 'xlnet', 'electra'
        :param saved_embedding_dir(str): path to the precomputed directory
        :param verbose (bool):
        :param embedding_batch_size (str): batch size to use when computed embeddings for a large data
        :param attention (bool): if True, the attention scores would be find the top topics/words within the cluster
        """
        self.verbose = verbose
        self.k = k
        self.dictionary = None
        self.corpus = None
        self.cluster_model = None
        self.basemodel = None
        self.vec = {}
        self.gamma = gamma  # parameter for relative importance of lda
        self.embedding = None
        self.base = base

        if embedding is None and base is None:
            print("Either enter a valid embedding or a valid base algorithm")
            return
        elif embedding is None:
            self.method = self.base
            self.id = (
                self.method
                + "_k_"
                + str(self.k)
                + "_"
                + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            )
        elif self.base is None:
            self.method = "EMBEDDING"
            self.embedding = embedding
            self.id = (
                self.method
                + "_k_"
                + str(self.k)
                + "_"
                + self.embedding
                + "_"
                + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            )
        else:
            self.method = self.base + "_EMBEDDING"
            self.embedding = embedding
            self.id = (
                self.method
                + "_k_"
                + str(self.k)
                + "_"
                + self.embedding
                + "_"
                + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            )

        self.dir = "output/{}/{}/".format(self.method, self.id)
        # create the directory for saving outputs
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.AE = None

        self.top_topics = None
        self.saved_embedding_dir = saved_embedding_dir
        self.saved_attention_dir = saved_attention_dir
        self.embedding_batch_size = embedding_batch_size
        self.use_AE = use_AE
        self.output_attention = output_attention

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vector representations from selected methods
        """

        if method == "LSI" or method == "LDA":

            print("Getting vector representations for {} ".format(self.base))

            # turn tokenized documents into a id <-> term dictionary
            self.dictionary = corpora.Dictionary(token_lists)
            if self.verbose:
                print("Dictionary used for {} \n".format(self.base))
                for key, value in self.dictionary.items():
                    print(key, value)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            if self.verbose:
                print("\n Corpus used for {} \n".format(self.base))
                for c in self.corpus:
                    print(c)

            if self.base == "LSI":
                self.basemodel = gensim.models.lsimodel.LsiModel(
                    self.corpus, num_topics=self.k, id2word=self.dictionary
                )
            elif self.base == "LDA":
                self.basemodel = gensim.models.ldamodel.LdaModel(
                    self.corpus, num_topics=self.k, id2word=self.dictionary, passes=20
                )

            def get_vec_base(model, corpus, k):
                """
                Get the Lsi vector representation (probabilistic topic assignments for all documents)
                :return: vec_lsi with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_base = np.zeros((n_doc, k))

                if self.base == "LSI":
                    for i in range(n_doc):
                        for topic, lsi_score in self.basemodel[self.corpus[i]]:
                            vec_base[i, topic] = lsi_score

                elif self.base == "LDA":

                    for i in range(n_doc):
                        # get the distribution for the i-th document in corpus
                        for topic, prob in model.get_document_topics(corpus[i]):
                            vec_base[i, topic] = prob

                return vec_base

            vec = get_vec_base(self.basemodel, self.corpus, self.k)
            if self.verbose:
                print("\n \n Shape of {} Vector: {}".format(self.base, vec.shape))
                print(vec)
                print(
                    "\n According to {}, the top words for each topic should be, ".format(
                        self.base
                    )
                )
                print(self.get_top_words_base(6))
            print("Getting vector representations for {}. Done! \n".format(self.base))
            self.vec[self.base] = vec
            return vec

        elif method == "EMBEDDING":

            print("\n Getting vector representations for {} ".format(self.embedding))

            if self.saved_embedding_dir is None:
                sentence_embedding, attention, embedding_tokens = get_embeddings(
                    self.embedding,
                    sentences=sentences,
                    batch_size=self.embedding_batch_size,
                    output_attention=self.output_attention,
                )
            else:
                # use previously generated embedding
                if self.saved_embedding_dir is not None:
                    sentence_embedding = get_embeddings(
                        saved_embedding_dir=self.saved_embedding_dir
                    )
                # use previously generated attention vector
                if self.saved_attention_dir is not None:
                    attention = get_embeddings(
                        saved_attention_dir=self.saved_attention_dir
                    )
                else:
                    # or generate a warning message
                    print("Attention Vector missing!")
                    attention = None
                    embedding_tokens = None

            vec = np.array(sentence_embedding)

            self.vec["EMBEDDING"] = vec
            self.vec["attention"] = attention
            self.vec["embedding_tokens"] = embedding_tokens
            return vec

        elif "_EMBEDDING" in method:

            # Get probability of a text belonging to a class from LDA/LSI
            vec_base = self.vectorize(sentences, token_lists, method=self.base)

            # Get sentence level embedding from Bert
            vec_embedding = self.vectorize(sentences, token_lists, method="EMBEDDING")

            # Concatenat them
            vec_base_embedding = np.c_[vec_base * self.gamma, vec_embedding]
            self.vec[method] = vec_base_embedding

            if self.verbose:
                print(
                    "\n Concatenating both {}(topic distribution per topic) and "
                    "{}(sentence embeddings) vectors gives a new vector of shape {}"
                    " \n".format(self.base, self.embedding, vec_base_embedding.shape)
                )

            return vec_base_embedding

    def fit_AE(self, vec, epoch):
        # If we want to use an autoencoder to learn latent representation
        if self.use_AE:

            # Learn the latent representation of data
            input_size = vec.shape[1]
            self.AE = Autoencoder(
                latent_dim=32, input_dim=input_size, epochs=epoch, batch_size=64
            )

            if self.verbose:
                print(
                    "Using the following autoencoder to learn the latent representation of the vector"
                )

            print("Fitting Autoencoder")

            self.AE.fit(vec)
            print("Fitting Autoencoder Done!")

        vec = self.AE.model.predict(vec)
        if self.verbose:
            print("Latent Representation size: ", np.array(vec).shape)
        if torch.cuda.is_available():
            return vec.cpu().numpy()
        else:
            return vec.numpy()

    def fit(self, sentences, token_lists, epoch=5, vis_clusters=False):
        """
        Fit the topic model for selected method given the preprocessed data

        :return:
        """

        # use kmeans clustering to cluster the latent representations
        self.lda_token_lists = token_lists
        m_clustering = KMeans
        self.cluster_model = m_clustering(self.k)

        self.vectorize(sentences, token_lists, self.method)
        if self.method == "LDA" or self.method == "LSI":
            print(" {} Done.".format(self.base))
            return  # no AE used

        if self.use_AE:  # used only if there is an embedding used
            self.vec["ENCODER"] = self.fit_AE(self.vec[self.method], epoch=epoch)
            print("Clustering the latent representations")
            self.cluster_model.fit(self.vec["ENCODER"])
        else:  # cluster the embedding representation
            self.cluster_model.fit(self.vec["EMBEDDING"])

        print("Clustering. Done!")
        print(
            "Silhouette Score for the clusters is",
            self.get_silhouette(plot=vis_clusters),
        )

    def get_top_words_base(self, top=5):

        if not self.basemodel:
            print("Base model has not been trained yet")
            return None

        self.top_topics = dict()

        if self.base == "LDA":
            for k in range(self.k):
                topics = self.basemodel.show_topic(topicid=k, topn=top)
                self.top_topics[k] = topics
        elif self.base == "LSI":
            for k in range(self.k):
                topics = self.basemodel.show_topic(topicno=k, topn=top)
                self.top_topics[k] = topics

        return self.top_topics

    def get_top_words_tfidf(
        self, sentence_label_dict, sentences=None, sentence_token=None, top=200
    ):
        """
        use tf-idf to find main topic words from clusters
        :param sentence_label_dict: (dict) containing the indices of the sentences belonging to each cluster
        :param sentences: list containing all sentences
        :param sentence_token: list of list containing all tokens
        :param top: int, top words to pick using tdidf score
        :return:
        """

        tfidf_sentences = []
        if sentence_token is not None:
            for i in range(self.k):
                # concatenate all tokens belong to cluster i into one string
                k_sentence = " ".join(
                    " ".join(word for word in sent)
                    for sent in np.array(sentence_token)[sentence_label_dict[i]]
                )
                # append to the list
                tfidf_sentences.append(k_sentence)
        elif sentences is not None:
            for i in range(self.k):
                # concatenate all sentences belong to cluster i into one string
                k_sentence = " ".join(
                    sent for sent in np.array(sentences)[sentence_label_dict[i]]
                )
                # remove the stop words
                # tokenize
                k_sentence = word_tokenize(k_sentence)
                # remove stopwords
                k_sentence = [
                    word for word in k_sentence if not word in stopwords.words()
                ]
                # convert back to string
                k_sentence = (" ").join(k_sentence)
                # append to the list
                tfidf_sentences.append(k_sentence)
        else:
            print("Either sentences or the token list must be provided")
            return
        tfidf = TfidfVectorizer()
        tfidf_vec = tfidf.fit_transform(tfidf_sentences)
        tfidf_features = tfidf.get_feature_names()

        pd_tfidf_vec = pd.DataFrame.sparse.from_spmatrix(tfidf_vec)
        tfidf_topics = []
        for k in range(self.k):
            k_topics = np.array(pd_tfidf_vec.iloc[[k]])
            # get the indices of the top k features, -ve sign return it in the descending order
            top_k_topic_ind = (-k_topics).argsort()[:, :top]

            # using the ids above, get feature names and add it to the list along with the score
            topic_k = dict()
            for i in top_k_topic_ind[0]:
                topic_k[tfidf_features[i]] = k_topics[0][i]
            tfidf_topics.append(topic_k)

        return tfidf_topics

    def get_top_words_using_attention(
        self, sentence_label_dict, cluster_id, layer=0, head=None, top=None
    ):

        if self.embedding not in ["xlnet", "bert", "xlm", "electra", "albert"]:
            print(
                " Attention vector is not available for the selected embedding model."
            )
            return
        attention_tokens = dict()
        token_list = []
        keys_to_remove = ["[CLS]", "[PAD]", "[SEP]"]

        attention_layers = self.vec["attention"]
        # attention vector for the samples belonging to this cluster
        cluster_attention = np.squeeze(
            attention_layers[:, [sentence_label_dict[cluster_id]], :, :, :]
        )
        # get the specified attention layer
        attention_layer = cluster_attention[layer]

        # get the tokens for all sentences belonging to this cluster
        indexed_tokens = np.array(self.vec["embedding_tokens"])[
            sentence_label_dict[cluster_id]
        ]

        for text_id, text in enumerate(attention_layer):
            tokens = indexed_tokens[text_id]
            tokens_NN = []
            if head is None:
                # iterate over all attention heads of the specified layer if head is not define
                for attention_head in text:

                    # iterate over all tokens and look for nouns
                    for token_id, token in enumerate(tokens):
                        # if the token is noun then sum over all the values in it's attention column, depicting all the attention paid
                        # to this token by all other tokens in the sentence
                        tag = nltk.pos_tag([token])
                        if (
                            tag[0][1] == "NN"
                            and token not in keys_to_remove
                            and ("#" not in token)
                        ):
                            tokens_NN.append(token)
                            print(token_id)
                            print(attention_head.shape)
                            attention_sum = np.sum(attention_head[:, token_id], axis=0)
                            if (
                                token in attention_tokens.keys()
                            ):  # if token already exists
                                attention_tokens[
                                    token
                                ] += attention_sum  # increment the value
                            else:
                                attention_tokens[token] = attention_sum
            else:  # if attention head is defined
                attention_head = text[head]
                attention_head = attention_head.numpy()
                # iterate over all tokens and look for nouns
                for token_id, token in enumerate(tokens):
                    # if the token is noun then sum over all the values in it's attention column, depicting all the attention paid
                    # to this token by all other tokens in the sentence
                    tag = nltk.pos_tag([token])
                    if (
                        tag[0][1] == "NN"
                        and token not in keys_to_remove
                        and ("#" not in token)
                    ):
                        tokens_NN.append(token)
                        attention_sum = np.sum(attention_head[:, token_id], axis=0)
                        if token in attention_tokens.keys():  # if token already exists
                            attention_tokens[
                                token
                            ] += attention_sum  # increment the value
                        else:
                            attention_tokens[token] = attention_sum
            token_list.append(tokens_NN)

        # sort in descending order
        attention_tokens = {
            k: v
            for k, v in sorted(
                attention_tokens.items(), key=lambda item: item[1], reverse=True
            )
        }
        # type cast attention scores to int
        attention_tokens = dict([a, int(x)] for a, x in attention_tokens.items())

        if top is not None:
            attention_tokens = Counter(attention_tokens).most_common(top)
            attention_tokens = dict(attention_tokens)
        return attention_tokens, token_list

    def compress_features(self, ids):
        """
        Preprocessing (Dimensionality Reduction) for visualization

            Perform Dimensionality reduction on the learned features using the following steps:

        :return: pandas dataframe with features vectors(averaged over all objectives for each employee),
        compress features vectors 2d. compressed feature vector 3d,
        assigned cluster label
        """

        # Getting feature vectors and cluster labels assigned by the model
        # Getting feature vectors and cluster labels assigned by the model
        if self.method == "LDA" or self.method == "LSI":
            feature_vector = self.vec[self.method]
            # for each sentence, find out the cluster it belongs to
            cluster_label_per_document = np.argmax(np.array(feature_vector), axis=1)
        else:
            cluster_label_per_document = np.array(self.cluster_model.labels_)
            feature_vector = self.vec[self.method]
            if torch.cuda.is_available():
                feature_vector = feature_vector.cpu()
            else:
                feature_vector = feature_vector

        num_employee = len(ids)

        # Create place holder for the analysis
        averaged_employee_features = pd.DataFrame(
            None,
            index=pd.Index(ids, name="Employee id"),
            columns=pd.Index(
                ["feature_vector", "cluster", "feature_vector_2d", "feature_vector_3d"],
                name="Compressed_features",
            ),
        )
        # get the indices of all objectives/documents/sentence for each employee (for later use)
        employee_objective_indices = dict()
        ids = np.array(ids)
        for employee in ids:
            employee_indices = [i for i, e in enumerate(ids) if (ids[i] == employee)]
            employee_objective_indices[employee] = employee_indices[0]

        #  putting values in the place holder

        for employee in ids:
            # get the feature vectors belonging to an employee
            averaged_employee_features["feature_vector"][employee] = feature_vector[
                employee_objective_indices[employee]
            ]

            # assign the predicted label for that employe
            averaged_employee_features["cluster"][
                employee
            ] = cluster_label_per_document[employee_objective_indices[employee]]

            # 3) doing dimensionality reduction into 2d/3d Space
        # todo: find a more permanent solution for this
        # panda returns a array of array for the column feature vector. Converting from array of array to 2d array
        feature_vector_ = np.stack(
            averaged_employee_features["feature_vector"].to_numpy()
        )

        # reduce dimension of feature vectors into 2
        feature_vector_2d = TSNE(n_components=2).fit_transform(feature_vector_)
        averaged_employee_features["feature_vector_2d"] = list(feature_vector_2d)

        # reduce dimension of feature vectors into 3
        feature_vector_3d = TSNE(n_components=3).fit_transform(feature_vector_)
        averaged_employee_features["feature_vector_3d"] = list(feature_vector_3d)

        return averaged_employee_features

    def wordcloud(self, sentences, ngram, output_directory=None):
        """
        Visualization Function
        This functions creates wordclouds as per the specified configurations through the user interface
        and saves them in the predefined directory
        :param TopicModelObject: TopicModel instances
        :return:
        """

        if output_directory is None:
            output_directory = self.dir

        if self.method == "LDA":
            # we use the word_distribution method for word cloud as we only have probability
            # of that word belonging to the cluster from LDA
            for k, cluster_word_distribution in self.get_top_words_base(
                top=200
            ).items():

                cluster_word_distribution_dict = dict()
                for item in cluster_word_distribution:
                    cluster_word_distribution_dict[item[0]] = item[1]

                get_wordcloud(
                    dir=output_directory,
                    word_distribution=cluster_word_distribution_dict,
                    topic=k,
                )
        else:

            # Getting indexes of sentences belonging to each cluster
            sentence_label_dict = dict()
            labels = self.cluster_model.labels_

            for label in set(labels):
                sentence_label_dict[label] = [
                    idx for idx, l in enumerate(labels) if l == label
                ]
            topic_words = []
            print("WordClouds for {} grams".format(str(ngram)))
            # get topic words for all clusters
            new_token_lists = []
            for i in range(self.k):
                k_sentences = np.array(sentences)[sentence_label_dict[i]]
                ngram_token_with_frequency, token_list = generate_n_gram(
                    k_sentences,
                    n=ngram,
                    separator="-",
                    keep_stopwords=False,
                    stem=True,
                    keep_nouns_only=True,
                )

                new_token_lists += token_list
                topic_words.append(ngram_token_with_frequency)

            topic_words = clean_topics_from_clusters(topics=topic_words, delete=False)

            # generate wordclouds
            for i in range(self.k):
                get_wordcloud(
                    topic=i, word_count_dict=topic_words[i], dir=output_directory
                )

    def get_labels(self):
        if self.method == "LDA" or self.method == "LSI":
            return np.argmax(self.vec[self.base], axis=1)
        else:
            return self.cluster_model.labels_

    def quick_stats(self, ids):
        """
        returns the distribution of employees in each cluster
        :param ids:
        :param cluster_label_per_document:
        :return:
        """

        unique_ids = list(set(ids))

        # get cluster prediction for all employees
        if self.method == "LDA":
            feature_vector = self.vec[self.method]
            # for each sentence, find out the cluster it belongs to
            cluster_label_per_document = np.argmax(np.array(feature_vector), axis=1)
        else:
            cluster_label_per_document = np.array(self.cluster_model.labels_)
        # create a dataframe to store the frequence per cluster for each employee.
        df_id_cluster = pd.DataFrame(
            0,
            index=pd.Index(unique_ids, name="Employee id"),
            columns=pd.Index([i for i in range(self.k)], name="Clusters"),
        )

        # There are multiple entries per employee so an employee can belong to multiple clusters
        for i, _id in enumerate(ids):
            df_id_cluster[cluster_label_per_document[i]][_id] = 1

        # shows how many employees belong to each cluster
        employee_cluster_frequency = np.sum(df_id_cluster.to_numpy(), axis=0)

        # shows percentage of employees belong to each cluster
        employee_cluster_frequency = (
            employee_cluster_frequency / len(unique_ids)
        ) * 100
        print(employee_cluster_frequency)
        create_hist(employee_cluster_frequency)

    def get_coherence(self, topics=None, token_lists=None, measure="c_v", top=100):
        """
        Score used for Topic Modeling
        Get model coherence from gensim.models.coherencemodel is a measure within-topic top word similiarity
        :param model: Topic_Model object
        :param token_lists: token lists of docs
        :param topics: topics as top words
        :param measure: coherence metrics
        :return: coherence score
        """

        def get_topic_words(token_lists, labels, k=None, top=100):
            """
            get top words within each topic from clustering results

            """
            if k is None:
                k = len(np.unique(labels))
            topics = ["" for _ in range(k)]
            for i, c in enumerate(token_lists):
                topics[labels[i]] += " " + " ".join(c)
            word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
            # get sorted word counts
            word_counts = list(
                map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts)
            )
            # get topics
            topics = list(
                map(lambda x: list(map(lambda x: x[0], x[:top])), word_counts)
            )

            return topics

        if self.method == "LDA":
            cm = CoherenceModel(
                model=self.basemodel,
                texts=self.lda_token_lists,
                corpus=self.corpus,
                dictionary=self.dictionary,
                coherence=measure,
            )
        else:

            if token_lists is None:
                print(
                    "Parameters missing, cannot compute coherence score for the model"
                )
                return
            dictionary = corpora.Dictionary(token_lists)

            # convert tokenized documents into a document-term matrix
            corpus = [dictionary.doc2bow(text) for text in token_lists]
            if topics is None:
                topics = get_topic_words(
                    token_lists, self.cluster_model.labels_, top=top
                )
            cm = CoherenceModel(
                topics=topics,
                texts=token_lists,
                corpus=corpus,
                dictionary=dictionary,
                coherence=measure,
            )

        return cm.get_coherence()

    def get_silhouette(self, plot=False):
        """
        Score used for clustering
        Get silhouette score from model. Not applicable for LDA

        Silhouette refers to a method of interpretation and validation of consistency within clusters of data.
        The technique provides a succinct graphical representation of how well each object has been classified.
        The silhouette value is a measure of how similar an object is to its own cluster compared to other clusters
        :return: silhouette score

        The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
        Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
        """
        if self.method == "LDA":
            print("Silhouette Score doesn't work with LDA")
            return

        lbs = self.get_labels()
        if self.use_AE:
            vec = self.vec["ENCODER"]
        else:
            vec = self.vec[self.method]

        score = silhouette_score(vec, lbs)
        if plot is False:
            return score

        print("Silhouette Score before dimensionality reduction is ", score)

        # performing dimensionality reduction to visualize the clusterings
        vec_3d = TSNE(n_components=3).fit_transform(vec)
        score = silhouette_score(vec_3d, lbs)
        print(
            "Silhouette Score after dimensionality reduction to a 3d space is ", score
        )

        # visualizing the clusters
        vis_3d(vec_3d, lbs)

        vec_2d = TSNE(n_components=2).fit_transform(vec)
        score = silhouette_score(vec_3d, lbs)
        print(
            "Silhouette Score after dimensionality reduction to a 2d space is ", score
        )

        # visualizing the clusters
        vis_2d(vec_2d, lbs)

        return score
