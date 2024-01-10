import openai
from openai.embeddings_utils import distances_from_embeddings
import numpy as np


class EmbeddingModel:
    def __init__(
        self,
        threshold=0.5,
        distance_metric="cosine",
        openai_embedding_model="text-embedding-ada-002",
    ):
        self.threshold = threshold
        self.distance_metric = distance_metric
        self.openai_embedding_model = openai_embedding_model
        self.type_error = "Invalid parameter"

    def get_embedding(self, text=None):
        try:
            return openai.Embedding.create(
                input=text, engine=self.openai_embedding_model
            )["data"][0]["embedding"]
        except TypeError as e:
            error_message = self.type_error + str(e)
            raise TypeError(error_message)

    def search(self, question=None, data_embeddings=None):
        try:
            question_embedding = self.get_embedding(question)
            distances = self.__get_distances(question_embedding, data_embeddings)
            indexes_distances_sorted = np.argsort(distances)
            indexes_most_similars = indexes_distances_sorted[
                distances[indexes_distances_sorted] < self.threshold
            ]
            return indexes_most_similars
        except TypeError as e:
            error_message = self.type_error + str(e)
            raise TypeError(error_message)

    def __get_distances(self, question_embedding=None, data_embeddings=None):
        try:
            return np.array(
                distances_from_embeddings(
                    question_embedding,
                    data_embeddings.values,
                    distance_metric=self.distance_metric,
                )
            )
        except TypeError as e:
            error_message = self.type_error + str(e)
            raise TypeError(error_message)
