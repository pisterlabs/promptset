import json
import os
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import click

import openai
import chromadb
from dotenv import load_dotenv


class IngestQasper:
    def __init__(self, qasper_path: str, chroma_dir: str):
        if not os.path.exists(qasper_path):
            raise ValueError(f"{qasper_path} is not exist")
        self.data = self.load_qasper(qasper_path)
        if not os.path.exists(chroma_dir):
            os.makedirs(chroma_dir)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection_name = 'qasper'
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                               model_kwargs={"device": "cuda"})

    def load_qasper(self, qasper_path: str):
        with open(qasper_path, 'rb') as r:
            data = json.load(r)
        return data

    def ingest(self):
        documents = []
        metadatas = []
        ids = []
        for idx, doi in enumerate(tqdm(list(self.data.keys()))):
            full_text = self.data[doi]['full_text']
            for text in full_text:
                section_name = text['section_name']
                paragraphs: List[str] = text['paragraphs']
                for i, paragraph in enumerate(paragraphs):
                    paragraph_id = f'{doi}_{section_name}_{i}'
                    documents.append(paragraph)
                    ids.append(paragraph_id)
                    assert doi is not None
                    if section_name is None:
                        section_name = 'Empty Section Name'
                    assert section_name is not None
                    metadatas.append({'doi': doi, 'section_name': section_name})

        embeddings = self.embedding.embed_documents(texts=documents)
        assert len(documents) == len(ids) == len(metadatas) == len(embeddings)

        # save to chroma
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )

    @staticmethod
    def embed_openai(sentence: str) -> List[float]:
        response = openai.Embedding.create(
            model='text-embedding-ada-002',
            input=sentence
        )
        return response['data'][0]['embedding']


@click.command()
@click.option('--qasper_path', type=str, required=True, default='./data/qasper/qasper-dev-v0.3.json')
@click.option('--chroma_dir', type=str, required=True, default='./Chroma')
def main(qasper_path: str, chroma_dir: str):
    IngestQasper(qasper_path, chroma_dir).ingest()


if __name__ == '__main__':
    main()
