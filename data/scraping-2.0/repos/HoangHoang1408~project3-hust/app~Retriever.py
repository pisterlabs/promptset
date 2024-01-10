import json
import os

import numpy as np
import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# load embeddings
model_name = "intfloat/multilingual-e5-large"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedder = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

defaut_search_config = {
    "k": 1,
    "score_threshold": 0.2,
}


def extract_title(x):
    for i in x.split("\n"):
        if "Tên thủ tục" in i:
            name = i.split(":")[1].strip()
            return name
    return None


class Retriever:
    def __init__(
        self, text_list, embedder, main_tokenizer, search_config=defaut_search_config
    ):
        self.text_list = text_list
        self.embedder = embedder
        self.main_tokenizer = main_tokenizer
        self.search_config = search_config
        self.corpus = None
        self.db = None
        self.retriever = None

    def _build_corpus(self, num_token):
        print("Building corpus...")
        print(f"Splitting {len(self.text_list)} documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=num_token,
            length_function=lambda x: len(self.main_tokenizer.tokenize(x)),
            chunk_overlap=0,
        )
        self.corpus = []
        for i, text in enumerate(self.text_list):
            title = extract_title(text)
            title = f"Tiêu đề: {title}\n" if title is not None else ""
            text_chunks = text_splitter.split_text(text)
            text_chunks = [title + chunk for chunk in text_chunks]
            text_docs = [
                Document(page_content=chunk, metadata={"id": i})
                for chunk in text_chunks
            ]
            self.corpus.extend(text_docs)

    def build(self):
        self._build_corpus()
        print(f"Embedding {len(self.corpus)} chunks...")
        self.db = FAISS.from_documents(self.corpus, self.embedder)
        self.retriever = self.db.as_retriever(search_kwargs=self.search_config)

    def save_local(self, path):
        os.makedirs(path, exist_ok=True)
        text_list_df = pd.DataFrame(self.text_list)
        corpus_list = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.corpus
        ]
        corpus_df = pd.DataFrame(corpus_list)
        text_list_df.to_csv(os.path.join(path, "text_list.csv"))
        corpus_df.to_csv(os.path.join(path, "corpus.csv"))
        self.db.save_local(os.path.join(path, "db"))
        with open(os.path.join(path, "search_config.json"), "w") as f:
            json.dump(self.search_config, f)

    @staticmethod
    def load_local(path, embedder, main_tokenizer):
        # check all paths exist
        paths = [
            os.path.join(path, "text_list.csv"),
            os.path.join(path, "corpus.csv"),
            os.path.join(path, "db"),
            os.path.join(path, "search_config.json"),
        ]
        for temp in paths:
            if not os.path.exists(temp):
                raise ValueError(f"Path {temp} does not exist")

        # load all files
        with open(os.path.join(path, "search_config.json"), "r") as f:
            search_config = json.load(f)
        text_list_df = pd.read_csv(os.path.join(path, "text_list.csv"))
        corpus_df = pd.read_csv(os.path.join(path, "corpus.csv"))
        text_list = text_list_df["0"].tolist()
        corpus = [
            Document(page_content=row["page_content"], metadata=eval(row["metadata"]))
            for _, row in corpus_df.iterrows()
        ]

        # load db
        db = FAISS.load_local(os.path.join(path, "db"), embedder)
        retriever = Retriever(
            text_list,
            embedder,
            main_tokenizer,
            search_config=search_config,
        )
        retriever.corpus = corpus
        retriever.db = db
        retriever.retriever = db.as_retriever(
            search_kwargs=retriever.search_config,
            search_type="similarity_score_threshold",
        )
        return retriever

    def search_main_document(self, query):
        result = self.retriever.get_relevant_documents(query)
        if len(result) == 0:
            return None
        candidate_doc = result[0]
        id_ = candidate_doc.metadata["id"]
        candidate_chunks = [
            doc.page_content for doc in self.corpus if doc.metadata["id"] == id_
        ]
        temp_embeddings = self.embedder.embed_documents(
            [doc for doc in candidate_chunks]
        )
        return {
            "id": id_,
            "chunk_texts": candidate_chunks,
            "chunk_embeddings": temp_embeddings,
        }

    def search_chunks(self, main_doc, query, k=2):
        if len(main_doc["chunk_texts"]) <= k:
            return {
                "id": main_doc["id"],
                "chunk_texts": main_doc["chunk_texts"],
                "chunk_socres": [1] * len(main_doc["chunk_texts"]),
            }
        q_embedding = self.embedder.embed_query(query)
        chunk_texts, chunk_embeddings = (
            main_doc["chunk_texts"],
            main_doc["chunk_embeddings"],
        )
        scores = np.dot(chunk_embeddings, q_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_embedding)
        )
        top_k = np.argsort(scores)[::-1][:k]
        return {
            "id": main_doc["id"],
            "chunk_texts": [chunk_texts[i] for i in top_k],
            "chunk_socres": [scores[i] for i in top_k],
        }
