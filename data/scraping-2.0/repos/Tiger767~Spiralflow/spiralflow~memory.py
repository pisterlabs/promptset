import numpy as np
import pandas as pd
import faiss
from typing import Dict, Optional
import pickle
from openai.embeddings_utils import get_embedding


def combine_on_overlap(str1: str, str2: str, threshold: float) -> Optional[str]:
    """
    Combines two strings if they overlap by a certain threshold.
    :param str1: First string to combine.
    :param str2: Second string to combine.
    :param threshold: Threshold for ratio of overlap to combine results from multiple queries.
    :return: Combined string if they overlap by a certain threshold, otherwise None.
    """

    max_overlap = min(len(str1), len(str2))
    best_overlap = 0
    best_overlap_index = -1
    overlap_type = None

    # Check for overlaps at the end of str1 and the beginning of str2
    for i in range(1, max_overlap + 1):
        if str1[-i:] == str2[:i] and i / len(str2) > threshold:
            if i > best_overlap:
                best_overlap = i
                best_overlap_index = i
                overlap_type = "end_start"

    # Check for overlaps at the beginning of str1 and the end of str2
    for i in range(1, max_overlap + 1):
        if str1[:i] == str2[-i:] and i / len(str1) > threshold:
            if i > best_overlap:
                best_overlap = i
                best_overlap_index = i
                overlap_type = "start_end"

    if best_overlap_index != -1:
        if overlap_type == "end_start":
            return str1 + str2[best_overlap_index:]
        elif overlap_type == "start_end":
            return str2 + str1[best_overlap_index:]
    else:
        return None


class Memory:
    def __init__(
        self,
        filepath: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
    ) -> None:
        """
        Initializes the memory.
        :param filepath: Path to a pickle file to load and save the memory to.
                         If None, the memory is created with text and metadata fields.
        :param embedding_model: Model to use for the embedding.
        """
        self.embedding_model = embedding_model

        if filepath is None:
            self.data = pd.DataFrame(columns=["text", "metadata"])
            self.index = None
        else:
            self.filepath = filepath
            self.load()

    def _create_index(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype(np.float32)
        _, d = embeddings.shape
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings / np.linalg.norm(embeddings, axis=-1))

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Saves the memory to a file.
        :param filepath: Path to the pickle file to save the memory to. If None, the filepath passed in the constructor is used.
        """
        if filepath is None:
            filepath = self.filepath

        with open(filepath + ".pkl", "wb") as f:
            pickle.dump(self, f)

    def load(self, filepath: Optional[str] = None) -> None:
        """
        Loads the memory from a pickle file.
        :param filepath: Path to a pickle file to load the memory from. If None, the filepath passed in the constructor is used.
        """
        if filepath is None:
            filepath = self.filepath

        with open(filepath, "rb") as f:
            loaded_memory = pickle.load(f)
            self.data = loaded_memory.data
            self.index = loaded_memory.index
            self.embedding_model = loaded_memory.embedding_model

    def add(
        self, data: Dict[str, str], save: bool = False, filepath: Optional[str] = None
    ) -> None:
        """
        Adds data to memory.
        :param data: Dict of data with a text and metadata field to add to memory.
        :param save: Whether to save the memory to a file.
        :param filepath: Path to the file (csv or parquet) to save the memory to.
                         If None, the filepath passed in the constructor is used.
        """

        if "text" not in data:
            raise ValueError("Data must have a 'text' field.")

        embedding = np.array(
            [get_embedding(data["text"], engine=self.embedding_model)], dtype=np.float32
        )

        data = pd.DataFrame(data, index=[0])
        self.data = pd.concat([self.data, data], ignore_index=True)

        if self.index is None:
            self._create_index(embedding)
        else:
            self.index.add(embedding / np.linalg.norm(embedding, axis=-1))

        if save:
            self.save(filepath)

    def query(
        self, query: str, k: int = 1, combine_threshold: Optional[float] = None
    ) -> list[Dict[str, str]]:
        """
        Queries the memory with the given query.
        :param query: Query to use to get memory.
        :param k: Max number of results to return.
        :param combine_threshold: Threshold for ratio of overlap to combine results from multiple queries.
                                  If None, no combining is done.
        :return: Memory obtained from external memories with metadata and scores (cosine similarity).
        """
        if self.data.empty:
            raise ValueError(
                "No memory to query. Add data to memory by calling Memory.add() before querying."
            )

        # get embedding of query
        embeded_query = np.array(
            [get_embedding(query, engine=self.embedding_model)], dtype=np.float32
        )

        # search for the indexes with similar embeddings
        scores, similar_indexes = self.index.search(
            embeded_query / np.linalg.norm(embeded_query, axis=-1), k=k
        )

        # get the memory values at the found indexes
        memories = []

        for score, i in zip(scores[0], similar_indexes[0]):
            memory = {
                "text": self.data.iloc[i, 0],
                "metadata": self.data.iloc[i, 1],
                "score": score,
            }
            memories.append(memory)

        if combine_threshold is not None and len(memories) > 1:
            # Need to improve algorithm as may miss some overlaps
            combined_memories = []
            for i in range(len(memories)):
                for j in range(len(combined_memories)):
                    combined_doc = combine_on_overlap(
                        combined_memories[j]["text"],
                        memories[i]["text"],
                        combine_threshold,
                    )
                    if combined_doc is not None:
                        combined_memories[j]["text"] = combined_doc
                        combined_memories[j]["score"] = min(
                            combined_memories[j]["score"], memories[i]["score"]
                        )
                        break
                else:
                    combined_memories.append(memories[i])
            memories = combined_memories

        return memories
