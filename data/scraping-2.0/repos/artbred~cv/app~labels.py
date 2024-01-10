import numpy as np
import openai
from settings import model_embeddings, openai_key, os
from storage import load_label_by_id, query_most_similar_labels, save_label_to_redis

openai.api_key = openai_key


def calculate_embedding(text: str, model=model_embeddings) -> np.array:
    text = text.replace("\n", " ").lower()
    embedding = openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)


class Label:
    def __init__(self, id, position, path_to_file, file_name=None, from_redis=True, *args, **kwargs):
        self.id = id
        self.position = position
        self.path_to_file = path_to_file
        self.file_name = os.path.basename(self.path_to_file) if file_name is None else file_name
        self.embedding = None

        # TODO parse embedding
        if not from_redis:
            self.embedding = calculate_embedding(self.position)

    def save_to_redis(self):
        label_hash = {
            "id": self.id,
            "position": self.position,
            "path_to_file": self.path_to_file,
            "file_name": self.file_name,
            "embedding": self.embedding.tobytes(),
        }

        save_label_to_redis(label_hash)


def get_most_similar_label(query_embedding: np.array) -> tuple[Label, float]:
    labels_document = query_most_similar_labels(query_embedding)
    score = -1

    if len(labels_document) == 0:
        return None, score

    most_similar_label_dict = labels_document[0].__dict__

    distance = most_similar_label_dict.get("dist")
    score = 1 - float(distance)

    return Label(**most_similar_label_dict), score


def get_label_by_id(id) -> Label:
    label_hash = load_label_by_id(id)

    if not label_hash:
        return None

    return Label(**label_hash)
