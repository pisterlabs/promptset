import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydash
import streamlit as st
from config import config
# from langchain.embeddings import VertexAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from networkx.algorithms import community
from scipy.spatial.distance import cosine
from utilities.custom_logger import CustomLogger

logger = CustomLogger()


@dataclass
class ProcessingPipeline:
    embeddings: Embeddings
    total_num_of_tokens: Optional[int] = None

    def process_document(self, document: str) -> List[str]:
        """process a long document into list of shorter chunks, where each chunk has a unique topic

        Args:
            document (str): long text

        Returns:
            List[str]: list of chunks
        """
        # remove sub-headers
        document = "\n\n".join([p if self.is_paragraph(p) else "\n\n" for p in document.split("\n\n")])

        self.total_num_of_tokens = self.get_num_of_tokens(document)

        # split documents into chunks based on its original paragraphing
        chunks = [ch for ch in re.split(r"[\n]{3,}", document) if len(ch) > 0]
        chunks = [config.remove_index(chunk, "[\\n]{2,}", "(\\d+ *)", "\n\n") for chunk in chunks]

        # ensure No. of tokens in each chunk < max context window
        chunks_require_split = list()
        for i, chunk in enumerate(chunks):
            chunk_len = self.get_num_of_tokens(chunk)
            logger.info(f"number of tokens for {i}th chunk is: {chunk_len}")
            if chunk_len > config.CHUNK_SIZE:
                chunks_require_split.append(i)

        if len(chunks_require_split) == 0:
            return chunks

        logger.info(f"long chunks require to be splitted are: {chunks_require_split}")

        apply_clsutering = True

        selected_chunks = []
        for i in chunks_require_split:
            if not apply_clsutering:
                text_splitter = CharacterTextSplitter().\
                    from_huggingface_tokenizer(
                    config.TOKENIZER,
                    chunk_size=config.CHUNK_SIZE // 2,
                    chunk_overlap=0,
                    separator="\n\n",
                    keep_separator=True
                )
                logger.info(f"partition the {i}th chunk due to large size:")
                splits = text_splitter.split_text(chunks[i])
                logger.info(f"partitioned long chunk into {len(splits)} sub-chunks")
                selected_chunks.extend(splits)

            else:
                logger.info(f"partition the {i}th chunk due to large size:")
                segments = chunks[i].split("\n\n")
                if len(segments) <= 2:
                    if len(segments) < 2:
                        raise Exception("paragraph - {i} is too long and cannot be split")
                    selected_chunks.extend(segments)
                else:
                    selected_chunks.extend(self.partition_segments(segments))
                    logger.info(f"partition for the {i}th chunk is completed")

        chunks = selected_chunks

        length_max = max([self.get_num_of_tokens(ch) for ch in chunks])
        length_min = min([self.get_num_of_tokens(ch) for ch in chunks])

        logger.info(f"After splitting by paragrah:\ntotal No. of chunks: {len(chunks)}, max length: {length_max}, min length: {length_min}")

        return chunks

    @staticmethod
    def get_num_of_tokens(text: str) -> int:
        """get No. of tokens in the text"""
        return len(config.TOKENIZER.tokenize(text))

    def is_paragraph(self, txt):
        """filter the paragraph with index"""
        if (re.match(r"^[0-9]+ ", txt) is None) and (self.get_num_of_tokens(txt) < 20):
            return False
        else:
            return True

    def partition_segments(self, segments: List[str]) -> List[str]:
        """To partition a large chunk, we first split the chunk into paragraphs, each paragraph is called a segment here. then
        we follow the algorithms below:
            1. search for any segment with length > config.CHUNK_SIZE
            2. consider the long segment as the original splitting points
            3. aggregate the rest of segments by text embedding and Louvain
            Community Detection Algorithm
            4. return partitions with texts

        Args:
            segments (List[str]): list of paragraphs

        Returns:
            List[str]: list of texts with length < config.CHUNK_SIZE
        """
        long = []
        for i, sub in enumerate(segments):
            sub_len = self.get_num_of_tokens(sub)
            logger.info(f"length of {i}th sub-chunk is {sub_len}")
            if sub_len > config.CHUNK_SIZE:
                # TODO:
                pass
            elif sub_len > config.CHUNK_SIZE * config.SPLIT_RATIO:
                long.append(i)

        def group_similar_segments(segments: List[str]) -> List[Set[int]]:
            embedding_dict = self.get_embeddings(segments)
            long = self.cluster_similar_chunks(embedding_dict)

            return long

        chunks = list()
        if len(long) == 0:
            clsuters = group_similar_segments(segments)
            for idx, clu in enumerate(clsuters):
                chunk = "\n\n".join([segments[i] for i in clu])
                logger.info(f"after partitioning, the length of {idx}th sub-chunk is {self.get_num_of_tokens(chunk)}")
                chunks.append(chunk)
            return chunks

        res = list()  # create a list to contain partitions
        segs = set()  # create a set to contain the elements of a cluster
        for i in range(len(segments)):
            if i in long:
                # add the previous cluster
                start = min(segs)
                end = max(segs) + 1
                clusters_ = group_similar_segments(segments[start:end])
                clusters = list()
                ori = range(start, end)
                for clu in clusters_:
                    clusters.append([ori[i] for i in clu])
                res.extend(clusters)

                # add the existing singleton
                res.append([i])

                # reset segs
                segs = set()
            else:
                segs.add(i)

        if len(segs) != 0:
            start = min(segs)
            end = max(segs) + 1
            clusters_ = group_similar_segments(segments[start:end])
            # map
            clusters = list()
            ori = range(start, end)
            for clu in clusters_:
                clusters.append([ori[i] for i in clu])
            res.extend(clusters)

        # del segs, ori, clusters_

        # map indexes back to texts
        for idx, clu in enumerate(res):
            chunk = "\n\n".join([segments[i] for i in clu])
            logger.info(f"after partitioning, the length of {idx}th sub-chunk is {self.get_num_of_tokens(chunk)}")
            chunks.append(chunk)

        return chunks

    def get_embeddings(self, paragraphs: List[str]) -> Dict[str, Dict]:
        """embeddings for each paragraph.
        The API accepts a maximum of 3,072 input tokens and outputs 768-dimensional vector embeddings.
        Use the following parameters for the text embeddings model textembedding-gecko(it belongs to PaLM Model)

        Args:
            paragraphs (List[str]): texts

        Returns:
            Dict[str, Dict]: embeddings
        """

        embedding_dict = dict()
        for idx, para in enumerate(paragraphs):
            sen_embedding = self.embeddings.embed_query(para)

            embedding_dict[str(idx)] = {
                "text": para,
                "embedding": sen_embedding
                }
        logger.info("embedding completed")
        # with open(config.OUT_PATH / "embedding_paragraph.json", "w") as f:
        #     json.dump(embedding_dict, f, indent=2)

        # load embeddings and get the similarity matrix for assessment
        # with open(config.OUT_PATH / "embedding_paragraph.json", "r") as f:
        #     embedding_dict = json.load(f)

        return embedding_dict

    def cluster_similar_chunks(self, embedding_dict: Dict[str, Dict]) -> List[Set[int]]:
        """
        cluster chunks into 1 if they share similar semantic meaning
        Args:
            embedding_dict (Dict[str, Dict]): embeddings

        Returns:
            List: list of chunk indexes
        """
        # Get similarity matrix between the embeddings of the sentences' embeddings
        summary_similarity_matrix = np.zeros((len(embedding_dict), len(embedding_dict)))
        summary_similarity_matrix[:] = np.nan

        for row in range(len(embedding_dict)):
            for col in range(row, len(embedding_dict)):
                # Calculate cosine similarity between the two vectors
                similarity = 1 - cosine(embedding_dict[str(row)]["embedding"], embedding_dict[str(col)]["embedding"])
                summary_similarity_matrix[row, col] = similarity
                summary_similarity_matrix[col, row] = similarity

        plt.figure()
        plt.imshow(summary_similarity_matrix, cmap='Blues')
        plt.savefig(config.OUT_PATH / "similarity_matrix_paragraph.jpg")

        partitions = self.get_topics(
            [t["text"] for t in embedding_dict.values()],
            summary_similarity_matrix,
            bonus_constant=0.2)

        # with open(config.OUT_PATH / "chunks", "wb") as fp:
        #     pickle.dump(chunks, fp)

        return partitions

    def get_topics(self,
                   texts: List[str],
                   similarity_matrix: np.ndarray,
                   bonus_constant: float = 0.25) -> List[Set[int]]:
        """calculate if chunks belong to same cluster based on louvain community detection algorithm

        Args:
            similarity_matrix (np.ndarray): cosine similarity between chunks
            num_topics (int, optional): number of chunks in the end. Defaults to 8.
            bonus_constant (float, optional): Defaults to 0.25. This adds additional similarity score
            to the embedding's consine similarity if the two sentences are near. The purpose is to encourage contiguous clustering.

        Returns:
            _type_: _description_
        """
        proximity_bonus_arr = np.zeros_like(similarity_matrix)
        for row in range(proximity_bonus_arr.shape[0]):
            for col in range(proximity_bonus_arr.shape[1]):
                if row == col:
                    proximity_bonus_arr[row, col] = 1
                else:
                    proximity_bonus_arr[row, col] = 1/(abs(row-col)) * bonus_constant

        similarity_matrix += proximity_bonus_arr

        similarity_matrix = nx.from_numpy_array(similarity_matrix)

        # Store the accepted partitionings
        resolution = 0.01  # increase resolution will favour smaller community
        resolution_step = 0.001
        partitions = []

        def is_partition_correct(partitions: List[Set], texts: List[str], thresh: int) -> bool:
            if len(partitions) == 0:
                return False

            for par in partitions:
                cluster = "\n\n".join([texts[p] for p in par])
                cluster_len = self.get_num_of_tokens(cluster)
                if cluster_len >= thresh:
                    return False

            return True

        while not is_partition_correct(partitions, texts, config.CHUNK_SIZE):
            partitions = community.louvain_communities(
                G=similarity_matrix,
                resolution=resolution,
                seed=1)
            resolution += resolution_step
        logger.info(f"successfully partitioned text into {partitions}")

        return partitions

# gcloud init:
# https://cloud.google.com/sdk/docs/initializing
# https://cloud.google.com/sdk/gcloud/reference/auth/activate-service-account#ACCOUNT
