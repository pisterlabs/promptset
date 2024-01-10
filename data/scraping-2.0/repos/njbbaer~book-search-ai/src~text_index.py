import json
import ast
from tqdm import tqdm
from openai.embeddings_utils import cosine_similarity

from src.util import create_embedding
from src.embedding_stats import print_stats


class TextIndex:
    def __init__(self, index):
        self.index = index

    @classmethod
    def build(cls, pages, print=False):
        if print:
            print_stats(pages)
        iterator = tqdm(range(len(pages)), disable=not print)
        index = [Item.create(i, pages[i]) for i in iterator]
        return cls(index)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as file:
            index = json.load(file)
        index = [Item.from_dict(page) for page in index]
        return TextIndex(index)

    def save(self, filepath):
        index = [page.to_dict() for page in self.index]
        with open(filepath, 'w') as file:
            json.dump(index, file, indent=2, ensure_ascii=False)

    def search(self, query):
        query_embedding = create_embedding(query)
        best_page = max(
            self.index,
            key=lambda item: cosine_similarity(query_embedding, item.embedding)
        )
        return best_page.text


class Item:
    def __init__(self, id, text, embedding):
        self.id = id
        self.text = text
        self.embedding = embedding

    @classmethod
    def create(cls, id, text):
        embedding = create_embedding(text)
        return cls(id, text, embedding)

    @classmethod
    def from_dict(cls, dct):
        return cls(
            dct["id"],
            dct["text"],
            ast.literal_eval(dct["embedding"])
        )

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'embedding': str(self.embedding)
        }
