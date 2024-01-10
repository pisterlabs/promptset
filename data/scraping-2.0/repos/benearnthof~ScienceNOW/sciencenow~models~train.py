"""
Wrapper class that unifies Topic Model training. 
"""
import json
import time
from pathlib import Path
from omegaconf import OmegaConf
from enum import Enum
from os import getcwd, listdir
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import warnings
from typing import Mapping, Any, List, Tuple


from river import cluster
from sklearn.feature_extraction.text import CountVectorizer

from cuml.cluster import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer, OnlineCountVectorizer

from sciencenow.data.arxivprocessor import ArxivProcessor
from sciencenow.utils.wrappers import Dimensionality, River, chunk_list

from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
import Levenshtein

# for manual coherence calculation
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import get_tmpfile


# run this in ScienceNOW directory
cfg = Path(getcwd()) / "./sciencenow/config/secrets.yaml"
config = OmegaConf.load(cfg)

class TM_PARAMS(Enum):
    """
    Wrapper class for Topic Model hyper parameters.
    Params:
        VOCAB_THRESHOLD:`int` that specifies how many times a word must occur in the corpus for 
            it to be contained in the topic model vocabulary. (This is separate from the vocab used for evaluation.)
    """
    VOCAB_THRESHOLD=config.VOCAB_THRESHOLD
    TM_VOCAB_PATH=config.TM_VOCAB_PATH
    TM_TARGET_ROOT=config.TM_TARGET_ROOT

setup_params = {
    "samples": 1, # hdbscan samples
    "cluster_size": 100, # hdbscan minimum cluster size
    "startdate": "01 01 2020", # if no date range should be selected set startdate to `None`
    "enddate": "31 01 2020",
    "target": "cs", # if no taxonomy filtering should be done set target to `None`
    "secondary_target": None, # for synthetic trend extraction
    "secondary_startdate": "01 01 2020",
    "secondary_enddate": "31 12 2020",
    "secondary_proportion": 0.1,
    "trend_deviation": 1.5, # value between 1 and 2 that determines how much more papers will be in the "trending bins"
                            # compared to the nontrending bins
    "n_trends": 1,
    "threshold": 0,
    "labelmatch_subset": None,  # if you want to compare results to another subset of data which may potentially 
                                # contain labels not present in the first data set this to a data subset.
    "mask_probability": 0,
    "recompute": True,
    "nr_topics": None,
    "nr_bins": 52, # number of bins for dynamic BERTopic, set to 52 for 52 weeks per year
    "nr_chunks": 12, # number of chunks the documents should be split up into for online learning, set to 52 for 52 weeks per year
    "evolution_tuning": False, # For dynamic model
    "global_tuning": False, # For dynamic model
    "limit": None,
}

online_params = {# For online DBSTREAM https://riverml.xyz/latest/api/cluster/DBSTREAM/
    "clustering_threshold": 1.0, # radius around cluster center that represents a cluster
    "fading_factor": 0.01, # parameter that controls importance of historical data to current cluster > 0
    "cleanup_interval": 2, # time interval between twwo consecutive time periods when the cleanup process is conducted
    "intersection_factor": 0.3, # area of the overlap of the micro clusters relative to the area cover by micro clusters
    "minimum_weight": 1.0 # minimum weight for a cluster to be considered not "noisy" 
}

class ModelWrapper():
    """
    Class to unify setup steps for Topic Model (tm) training. 
    Params:
        subset: `dataframe` that contains documents, timestamps and labels
        tm_params: `TM_PARAMS` enum that contains all hyperparameters
        setup_params: `Dict` with hyperparameters for model setup
        model_type: `str`; one of "base", "dynamic", "online", "antm" 
    """
    def __init__(
        self,
        tm_params=TM_PARAMS,
        setup_params=setup_params,
        online_params=online_params,
        model_type="base",
        ) -> None:
        super().__init__()
        #### Setting up data subset, labels & embeddings via processor
        self.processor = ArxivProcessor()
        self.processor.load_snapshot()
        self.tm_params = tm_params
        self.setup_params = setup_params
        self.online_params = online_params
        # loading subset returns None but assigns subsets
        self.load_subset()
        self.plaintext_labels, self.numeric_labels = self.processor.get_numeric_labels(
            subset = self.subset,
            mask_probability=self.setup_params["mask_probability"])
        if self.setup_params["limit"] is not None:
            self.subset = self.processor.reduce_subset(self.subset, limit=self.setup_params["limit"])
        model_types=["base", "dynamic", "semisupervised", "online", "antm", "embetter"]
        if model_type not in model_types:
            raise ValueError(f"Invalid model type. Expected on of {model_types}")
        self.model_type = model_type
        if self.model_type == "semisupervised": #recompute embeddings with supervised umap
            self.processor.bertopic_setup(
                subset=self.subset, recompute=True, labels=self.numeric_labels
                )
        else:
            self.processor.bertopic_setup(
                subset=self.subset, recompute=True
                )
        self.subset_reduced_embeddings = self.processor.subset_reduced_embeddings
        #### vocab for evaluation
        self.tm_vocab = None
        #### outputs
        self.topic_model = None
        self.topics = None
        self.probs = None
        self.topic_info = None
        self.topics_over_time = None
        #### Evaluation
        self.model_name = "BERTopic"
        self.topk = 5
        # prepare data and metrics
        self.data = self.get_dataset()
        self.corpus = self.data.get_corpus()
        self.metrics = self.get_metrics()
        self.verbose = True

    def load_subset(self):
        """
        Method that loads a data subset as specified by the setup parameters.
        If a cache directory is given, subsets will be written to disk at the corresponding
        location. If a subset exists in the specified location, this subset will be loaded instead.
        """
        if self.setup_params["subset_cache"] is not None:
            # TODO: replace with path variable
            Path(self.setup_params["subset_cache"]).mkdir(parents=True, exist_ok=True)
            filelist = listdir(Path(self.setup_params["subset_cache"]))
            subset_id = self.setup_params["startdate"].replace(" ", "") + self.setup_params["enddate"].replace(" ", "") + ".feather"
            secondary_subset_id = self.setup_params["secondary_startdate"].replace(" ", "") + self.setup_params["secondary_enddate"].replace(" ", "") + ".feather"
            merged = subset_id.split(".")[0] + secondary_subset_id
            # merged subset with fake trends
            if merged in filelist:
                print(f"Loading merged subset {merged} from cache...")
                self.subset = pd.read_feather(Path(self.setup_params["subset_cache"]) / merged)
            # regular subset with no fake data
            elif subset_id in filelist:
                print(f"Loading subset {subset_id} from cache...")
                self.subset = pd.read_feather(Path(self.setup_params["subset_cache"]) / subset_id)
            else:
                self.subset=self.processor.filter_by_date_range(
                startdate=self.setup_params["startdate"],
                enddate=self.setup_params["enddate"]
                )
                self.subset = self.processor.filter_by_taxonomy(
                    subset=self.subset, 
                    target=self.setup_params["target"], 
                    threshold=self.setup_params["threshold"]
                )
                if self.setup_params["secondary_target"] is not None:
                    self.secondary_subset = self.processor.filter_by_date_range(
                        startdate=self.setup_params["secondary_startdate"],
                        enddate=self.setup_params["secondary_enddate"]
                    )
                    self.secondary_subset = self.processor.filter_by_taxonomy(
                        subset=self.secondary_subset, 
                        target=self.setup_params["secondary_target"], 
                        threshold=0
                    )
                    self.subset = self.merge_subsets(proportion=self.setup_params["secondary_proportion"])
                    self.subset = self.subset.reset_index()
                    self.subset.to_feather(Path(self.setup_params["subset_cache"]) / merged)
                    print(f"Stored subset at {Path(self.setup_params['subset_cache']) / merged}.")
                # save
                self.subset = self.subset.reset_index()
                self.subset.to_feather(Path(self.setup_params["subset_cache"]) / subset_id)
                print(f"Stored subset at {Path(self.setup_params['subset_cache']) / subset_id}.")
        else: 
            warnings.warn("Please specify cache directory to store intermediate dataframes.")

    def merge_subsets(self, proportion=0.1):
        """
        Merge a subset of "real" data with a subset of "fake" papers from another category.
        This will serve as a setup function to simulate the influx of new papers
        """
        s1, s2 = self.subset, self.secondary_subset
        # simplest idea: Insert proportion of s2 into s1
        # only using papers belonging to one l1 class
        groups = Counter(s2.plaintext_labels.tolist()).most_common()
        target_amount = int(len(s1) * proportion)
        target_set = s2[s2.plaintext_labels==groups[0][0]]
        if len(target_set) > target_amount:
            target_set = target_set.sample(n = target_amount)
            print(f"Selected {target_amount} papers to be merged with subset.")
            target_set = self.adjust_timestamps(target_set)
        else:
            print(f"Selecting {target_amount} papers from multiple classes...")
            target_set = s2.sample(n = target_amount)
            target_set = self.adjust_timestamps(target_set)
        combined_set = pd.concat([s1, target_set])
        # preserve order of timestamps
        combined_set = combined_set.sort_values("v1_datetime")
        return combined_set

 
    def adjust_timestamps(self, target_set):
        """
        Method to adjust the timestamps of a "fake" dataset to fall in line with the
        timestamps of a target dataset.
        """
        target_timestamps = self.subset.v1_datetime.tolist()
        bins = self.setup_params["nr_bins"]
        n_trends = self.setup_params["n_trends"]
        deviation = self.setup_params["trend_deviation"]
        multipliers = [1] * (bins- n_trends) + [deviation] * n_trends
        random.shuffle(multipliers)
        # normalize so we can calculate how many papers should fall into each bin
        multipliers = np.array([float(i) / sum(multipliers) for i in multipliers])
        papers_per_bin = np.rint(multipliers * len(target_set)).astype(int)
        # make sure that samples will match in length
        new_set = target_set[0:sum(papers_per_bin)-1]
        new_timestamps = []
        binned_timestamps = np.array_split(target_timestamps, bins)
        for i, n in enumerate(papers_per_bin):
            sample = random.choices(binned_timestamps[i].tolist(), k=n)
            new_timestamps.extend(sample)
        new_timestamps = new_timestamps[0:len(new_set)]
        assert len(new_timestamps) == len(new_set)
        new_set.v1_datetime = new_timestamps
        return(new_set)


    def generate_tm_vocabulary(self, recompute=True):
        """
        Generate a vocabulary to offload this computation from tm training.
        Params:
            recompute: `Bool` indicating if vocabulary should be recomputed for subset of interest.
        """
        assert self.subset is not None
        if recompute:
            print(f"Recomputing vocab for {len(self.subset)} documents...")
            vocab = Counter()
            tokenizer = CountVectorizer().build_tokenizer()
            docs = self.subset.abstract.tolist()
            for doc in tqdm(docs):
                vocab.update(tokenizer(doc))
            reduced_vocab = [word for word, freq in vocab.items() if freq >= self.tm_params.VOCAB_THRESHOLD.value]
            print(f"Reduced vocab from {len(vocab)} to {len(reduced_vocab)}")
            with open(self.tm_params.TM_VOCAB_PATH.value, "w") as file:
                for item in reduced_vocab:
                    file.write(item+"\n")
            file.close()
            self.tm_vocab = reduced_vocab
            print(f"Successfully saved Topic Model vocab at {self.tm_params.TM_VOCAB_PATH.value}")
        else:
            self.load_tm_vocabulary()
        
    def load_tm_vocabulary(self):
        """Wrapper to quickly load a precomputed tm vocabulary to avoid recomputing between runs."""
        if not self.tm_params.TM_VOCAB_PATH.value.exists():
            warnings.warn(f"No tm vocabulary found at {self.tm_params.TM_VOCAB_PATH.value}.Consider calling `generate_tm_vocabulary` first.")
        else:
            with open(self.tm_params.TM_VOCAB_PATH.value, "r") as file:
                self.tm_vocab = [row.strip() for row in file]
            file.close()
            print(f"Successfully loaded TopicModel vocab of {len(self.tm_vocab)} items.")

    def tm_setup(self):
        """
        Wrapper to quickly set up a new topic model.
        """
        # use precomputed reduced embeddings 
        self.umap_model = Dimensionality(self.subset_reduced_embeddings)
        if self.model_type == "online":
            self.cluster_model = River(
                model=cluster.DBSTREAM(**self.online_params)
            )
            # set min_df to 5 to avoid running out of memory during coherence calc
            self.vectorizer_model = OnlineCountVectorizer(min_df=10, stop_words="english")
            # bm25_weighting helps to increase robustness to stop words in smaller online data chunks
            self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
            # need to recompute embeddings for online model
            self.umap_model = self.processor.umap_model
            print(f"Successfully set up online model with umap model: {self.umap_model.n_neighbors} neighbors.")
        else: 
            self.cluster_model = HDBSCAN( # TODO: add KMEANS & River for online & supervised models
                min_samples=self.setup_params["samples"],
                gen_min_span_tree=True,
                prediction_data=True,
                min_cluster_size=self.setup_params["cluster_size"],
                verbose=False,
            )
            self.vectorizer_model = CountVectorizer(min_df=10, stop_words="english")
            # remove stop words for vectorizer just in case
            self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.cluster_model,
            ctfidf_model=self.ctfidf_model,
            vectorizer_model=self.vectorizer_model,
            verbose=True,
            nr_topics=self.setup_params["nr_topics"],
            calculate_probabilities=False,
        )
        print("Setup complete.")

    def tm_train(self):
        """
        Wrapper to train topic model.
        """
        if self.topic_model is None:
            warnings.warn("No topic model set up yet. Call `tm_setup` first.")
        start = time.time()

        if self.model_type == "base":
            docs = self.subset.abstract.tolist()
            embeddings = self.subset_reduced_embeddings
            self.topics, _ = self.topic_model.fit_transform(documents=docs, embeddings=embeddings)
            self.topic_info = self.topic_model.get_topic_info()
            print("Base Topic Model fit successfully.")

        elif self.model_type in ["dynamic", "semisupervised"]:
            # Training procedure is the same since both models use timestamps and for semisupervised
            # the reduced embeddings were already recomputed based on the numeric labels while initializing
            # the class
            docs = self.subset.abstract.tolist()
            embeddings = self.subset_reduced_embeddings
            timestamps = self.subset.v1_datetime.tolist()
            self.topics, _ = self.topic_model.fit_transform(docs, embeddings)
            # reassign to hopefully help with topicover time representation
            # https://github.com/MaartenGr/BERTopic/issues/1593
            # merge models?
            self.topic_model.topics_ = self.topics
            print(f"Fitting dynamic model with {len(timestamps)} timestamps and {self.setup_params['nr_bins']} bins.")
            self.topics_over_time = self.topic_model.topics_over_time(
                docs, 
                timestamps, 
                nr_bins=self.setup_params["nr_bins"],
                evolution_tuning=setup_params["evolution_tuning"],
                global_tuning=setup_params["global_tuning"],
                )
            self.topic_info = self.topic_model.get_topic_info()
            
        elif self.model_type == "online": 
            # subset is already ordered by datetime
            doclist = self.subset.abstract.tolist()
            doc_chunks = chunk_list(doclist, n=setup_params["nr_chunks"])
            timestamps = self.subset.v1_datetime.tolist()
            self.topics = []
            for docs in tqdm(doc_chunks):
                self.topic_model.partial_fit(docs)
                self.topics.extend(self.topic_model.topics_)
            # for postprocessing
            self.topic_model.topics_ = self.topics
            # need to quantize into bins for trend extraction
            self.topics_over_time = self.topic_model.topics_over_time(
                doclist, 
                timestamps, 
                nr_bins=self.setup_params["nr_bins"],
                evolution_tuning=setup_params["evolution_tuning"],
                global_tuning=setup_params["global_tuning"],
                )
            self.topic_info = self.topic_model.get_topic_info()
            # self.output_tm = self.get_dynamic_topics(self.topics_over_time)

        elif self.model_type == "antm": # TODO: adapt from script
            pass
        elif self.model_type == "embetter": # TODO: adapt from script
            pass

        end = time.time()
        self.computation_time = float(end - start)
        return self.computation_time, self.topics
    
    def tm_save(self, name):
        """
        Wrapper to save a trained topic model.
        """
        if self.topic_model is not None:
            self.topic_model.save(
                path=Path(self.tm_params.TM_TARGET_ROOT.value) / name,
                serialization="safetensors",
                save_ctfidf=True
                )
            print(f"Model saved successfully in {self.tm_params.TM_TARGET_ROOT.value}")
        else:
            print(f"No Model found at specified location: {self.tm_params.TM_TARGET_ROOT.value}")

    def tm_load(self, name):
        """
        Wrapper to load a pretrained topic model.
        """
        assert (Path(self.tm_params.TM_TARGET_ROOT.value) / name).exists()
        self.topic_model = BERTopic.load(Path(self.tm_params.TM_TARGET_ROOT.value) / name)
        print(f"Topic model at {Path(self.tm_params.TM_TARGET_ROOT.value) / name}loaded successfully.")

    def get_dataset(self):
        """
        Get dataset in OCTIS format
        """
        data = Dataset()
        data.load_custom_dataset_from_folder(config.EVAL_ROOT)
        return data

    def get_metrics(self):
        """
        Prepare evaluation metrics using OCTIS
        """
        npmi = Coherence(texts=self.corpus, topk=self.topk, measure="c_npmi")
        #npmi = Coherence(topk=self.topk, measure="c_npmi")
        topic_diversity = TopicDiversity(topk=self.topk)
        # Define methods
        coherence = [(npmi, "npmi")]
        diversity = [(topic_diversity, "diversity")]
        # metrics = [(coherence, "Coherence"), (diversity, "Diversity")]
        # we will calculate coherence manually to avoid a problem with empty topics
        metrics = [(diversity, "Diversity")]
        return metrics

    def get_dynamic_topics(self, topics_over_time):
        """
        Helper method for evaluation of dynamic topic models.
        """
        unique_timestamps = topics_over_time.Timestamp.unique()
        dtm_topics = {} 
        all_words_cased = list(set([word for words in self.corpus for word in words]))
        all_words_lowercase = [word.lower() for word in all_words_cased]

        for unique_timestamp in tqdm(unique_timestamps):
            dtm_topic = topics_over_time.loc[
                topics_over_time.Timestamp == unique_timestamp, :
            ].sort_values("Frequency", ascending=True)
            dtm_topic = dtm_topic.loc[dtm_topic.Topic != -1, :]
            # TODO: investigate why we remove the "outlier" topic
            dtm_topic = [topic.split(", ") for topic in dtm_topic.Words.values]
            dtm_topics[unique_timestamp] = {"topics": dtm_topic}
            # replacement is in all words
            # all words contain capital letters, but topic labels only contain lower case letters
            updated_topics = []
            for topic in dtm_topic:
                updated_topic = []
                for word in topic:
                    if word not in all_words_cased and word not in all_words_lowercase:
                        # append word with minimal hamming distance from vocabulary
                        #distances = [Levenshtein.hamming(word.lower(), aword.lower()) for aword in all_words_cased]
                        #replacement = all_words_cased[np.argmin(distances)]
                        all_words_cased.append(word)
                        updated_topic.append(word)
                        #print(f"Word: {word} Replacement: {replacement}")
                        print(word)
                    else:
                        updated_topic.append(word)
                updated_topics.append(updated_topic)
            dtm_topics[unique_timestamp] = {"topics": updated_topics}
        return dtm_topics

    def get_base_topics(self):
        """
        Helper method for evaluation of base model.
        """
        all_words = [word for words in self.corpus for word in words]
        bertopic_topics = [
            [
                vals[0] if vals[0] in all_words else all_words[0]
                for vals in self.topic_model.get_topic(i)[:10]
            ]
            for i in range(len(set(topics)) - 1)
        ]
        output_tm = {"topics": bertopic_topics}
        return output

    def evaluate(self, output_tm):
        """
        Use predefined metrics and output of the topic model to evaluate the tm.
        """
        if self.model_type in ["base"]:
            results = {}
            for scorers, _ in self.metrics:
                for scorer, name in scorers:
                    score = scorer.score(output_tm)
                    results[name] = float(score)
            if self.verbose:
                print("Results")
                print("============")
                for metric, score in results.items():
                    print(f"{metric}: {str(score)}")
                print(" ")
        elif self.model_type in ["dynamic", "semisupervised", "online"]:
            results = {str(timestamp): {} for timestamp, _ in output_tm.items()}
            print(f"Evaluating Coherence and Diversity for {len(results)} timestamps.")
            for timestamp, topics in tqdm(output_tm.items()):
                self.metrics = self.get_metrics()
                for scorers, _ in self.metrics:
                    for scorer, name in scorers:
                        score = scorer.score(topics)
                        results[str(timestamp)][name] = float(score)
        return results

    def calculate_coherence(self):
        """
        Manually calculating topic coherence to avoid an OCTIS bug that happened with 
        empty topics.
        https://github.com/MaartenGr/BERTopic/issues/90
        """
        documents = pd.DataFrame({
            "Document": self.subset.abstract.tolist(),
            "ID": range(len(self.subset)),
            "Topic": self.topics
            })
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = self.topic_model._preprocess_text(documents_per_topic.Document.values)
        # use vectorizer with different min_df to reduce memory requirements
        vectorizer = self.topic_model.vectorizer_model
        #vectorizer = CountVectorizer(min_df=15, stop_words="english")
        analyzer = vectorizer.build_analyzer()
        words = vectorizer.get_feature_names()
        # Extract features for Topic Coherence evaluation
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        corpus_output_fname = get_tmpfile("corpus.mm")
        # save corpus as sparse matrix
        corpora.MmCorpus.serialize(corpus_output_fname, corpus)
        mm_corpus = corpora.MmCorpus(corpus_output_fname)
        # Extract words in each topic if they are non-empty and exist in the dictionary
        topic_words = []
        for topic in range(len(set(self.topics))-self.topic_model._outliers):
            words = list(zip(*self.topic_model.get_topic(topic)))[0]
            words = [word for word in words if word in dictionary.token2id]
            topic_words.append(words)
        topic_words = [words for words in topic_words if len(words) > 0]
        coherence_model = CoherenceModel(
            topics=topic_words, 
            texts=tokens, 
            corpus=mm_corpus,
            dictionary=dictionary, 
            coherence='c_v'
            )
        print(f"Calculating coherence for {len(Counter(self.topics))} topics...")
        coherence = coherence_model.get_coherence()
        return coherence


    def train_and_evaluate(self, save: str = False) -> Mapping[str, Any]:
        """
        Train a topic model, evaluate it, and return performance metrics.
        """
        results = []
        # setup topic model with specified parameters
        self.tm_setup()
        computation_time, topics = self.tm_train()
        print(f"Getting evaluation outputs for {self.topics_over_time.shape} Topics over time.")
        if self.model_type == "base":
            self.output_tm = self.get_base_topics()
        elif self.model_type in ["dynamic", "semisupervised", "online"]:
            self.output_tm = self.get_dynamic_topics(self.topics_over_time)

        print(f"Calculating Scores...")
        diversity = self.evaluate(self.output_tm)
        coherence = self.calculate_coherence()
        # update results 
        result = {
            "Dataset": "Arxiv",
            "Dataset Size": len(self.corpus),
            "Model": "BERTopic",
            "Params": self.setup_params,
            "OnlineParams": self.online_params,
            "Diversity": diversity,
            "Coherence": coherence,
            "Computation Time": computation_time,
            "Topics": topics,
        }
        results.append(result)
        if save:
            with open(f"{save}.json", "w") as f:
                json.dump(results, f)
        return results
