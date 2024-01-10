#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import logging
import os
import random
import tempfile

import gensim
import joblib
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from kolibri.cluster.baseTopic import TopicModel
from kolibri.pipeComponent import Component
from kolibri.settings import resources_path
from kolibri.utils.downloader import Downloader
from kolibri.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

TOPIC_MODEL_FILE_NAME = "topic_mallet_model.pkl"

PACKAGE = 'models/mallet'
DATA_DIR = resources_path
MALLET_DIR = os.path.join(DATA_DIR, PACKAGE)
URL_DATA="https://www.dropbox.com/s/lqcygea2y53rfbm/mallet-2.0.7.tar.gz?dl=1"


mallet_path = os.path.join(MALLET_DIR, 'mallet-2.0.7/bin/mallet')

class LdaMallet(Component, Downloader, TopicModel):

    """Python wrapper for LDA using `MALLET <http://mallet.cs.umass.edu/>`_.

    Communication between MALLET and Python takes place by passing around data files on disk
    and calling Java with subprocess.call().

    Warnings
    --------
    This is **only** python wrapper for `MALLET LDA <http://mallet.cs.umass.edu/>`_,
    you need to install original implementation first and pass the path to binary to ``mallet_path``.

    """

    name = "lda_topics"

    provides = ["topics"]

    requires = ["tokens"]

    defaults = {
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        "num_topics": 20,

        # The maximum number of iterations for optimization algorithms.
        "alpha": 50,
        "workers": 4,
        "optimize_interval": 0,
        "iterations": 200,
        "embeddings_dim":50,
        "topic_threshold":0.0,
        "random_seed": 0,
        "use_lemma": True,
        "nb_topic_start": 1,
        "nb_topic_stop": 1,
        "step": 1,
        "output_folder": "."

    }

    def __init__(self, component_config=None, vocabulary=None, prefix=None):


        Component.__init__(self, component_config)
        Downloader.__init__(self,
            PACKAGE,
            url=URL_DATA,
            download_dir=DATA_DIR)
        start=self.component_config["nb_topic_start"]
        stop=self.component_config["nb_topic_stop"]
        if  start > stop:
            raise Exception("In topic experimentation start should be larger than stop.")
        self.mallet_path = mallet_path
        self.vocabulary = vocabulary
        self.num_topics = self.component_config["num_topics"]
        self.topic_threshold = self.component_config["topic_threshold"]
        self.alpha = self.component_config["alpha"]
        if prefix is None:
            rand_prefix = hex(random.randint(0, 0xffffff))[2:] + '_'
            prefix = os.path.join(tempfile.gettempdir(), rand_prefix)
        self.prefix = prefix
        self.workers = self.component_config["workers"]
        self.optimize_interval = self.component_config["optimize_interval"]
        self.iterations = self.component_config["iterations"]
        self.random_seed = self.component_config["random_seed"]
        self.topic_model=None


    def train(self, training_data, cfg, **kwargs):
        """Train Mallet LDA.
        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format
        """

        if self.vocabulary is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.vocabulary = Vocabulary()
            self.vocabulary.add_training_data(training_data)

        else:
            self.num_terms = 0 if not self.vocabulary else 1 + max(self.vocabulary.keys())
        if len(self.vocabulary.vocab) == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")
        self.vocabulary.build()
        self.num_terms = self.vocabulary.count
        corpus = [self.vocabulary.doc2bow(doc) for doc in training_data.training_examples]

        if self.component_config["nb_topic_start"]-self.component_config["nb_topic_stop"]==0:
            self.topic_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, iterations=self.iterations, num_topics=self.num_topics, id2word=self.vocabulary.id2token)
        else:
            start=self.component_config["nb_topic_start"]
            limit=self.component_config["nb_topic_stop"]
            step=self.component_config["step"]
            texts=[]
            for example in training_data.training_examples:
                texts.append([t.text for t in example.tokens])
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step):
                model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, iterations=self.iterations, id2word=self.vocabulary.id2token)
                model_list.append(num_topics)
                coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=self.vocabulary, coherence='c_v')
                coherence_values.append(coherencemodel.get_coherence())

            x = range(start, limit, step)
            plt.plot(x, coherence_values)
            plt.xlabel("Num Topics")
            plt.ylabel("Coherence score")
            plt.legend(("coherence_values"), loc='best')
            plt.savefig(os.path.join(self.component_config["output_folder"], "coherence_plot.png"))

    def process(self, message, **kwargs):

        self._check_nlp_doc(message)
        message.set_output_property("topics")
        bow = [self.vocabulary.doc2bow(message)]

        message.topics = self.topic_model[bow]


    @classmethod
    def load(cls,
             model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs
             ):
        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("topic_file", TOPIC_MODEL_FILE_NAME)
        classifier_file = os.path.join(model_dir, file_name)

        if os.path.exists(classifier_file):
            model = joblib.load(classifier_file)

            return model
        else:
            return cls(meta)


    def persist(self, model_dir):
        """Persist this model into the passed directory."""

        classifier_file = os.path.join(model_dir, TOPIC_MODEL_FILE_NAME)
        joblib.dump(self, classifier_file)

        return {"topic_file": TOPIC_MODEL_FILE_NAME}
