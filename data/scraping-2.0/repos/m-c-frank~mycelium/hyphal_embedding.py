"""
hyphal nodes are the representations of any and or all information
they have their own representation
and they have child nodes
"""

from pathlib import Path
import numpy as np
from scipy.special import softmax

from annoy import AnnoyIndex
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import utils.cli as cli
import utils.helpers as helpers

DEFAULT_PATH_DATA = "data"
INDEX_TREES = 2
CHUNK_SIZE = 512  # 512 characters in document
CHUNK_OVERLAP = 64  # 64 characters overlap


class HyphalEmbedding:
    """
    a hyphal embedding is the representation of one single node as an embedding
    i.e. it has one latent representation
    and for now one text representation
    thats it.

    in the future it will have different representations
    but that will be handled by other classes
    like the different color channels of an image

    this is like the ultimate source of truth
    """

    content: str  # plain text representation of this hyphal
    embeddings: list[list[float]]  # embedding vector of this hyphal document

    def __init__(self, content: str, embeddings: list[list[float]]):
        """
        initializes the node
        embeds the __repr__ of a hyphal node

        this can later be optimized by rephrasing the chunks
        to be more closely related to the target
        this can be done by the functionality i outlined in the note repository
        """
        self.content = content
        self.embeddings = embeddings

    def __str__(self) -> str:
        return self.content

    @staticmethod
    def from_hyphal(hyphal: Path) -> "HyphalEmbedding":
        print(f"workin on hyphal {hyphal}")
        content, chunks = "", []
        if hyphal.suffix == ".pdf":
            content, chunks = HyphalEmbedding.from_pdf(hyphal)
        if hyphal.suffix == ".md":
            content, chunks = HyphalEmbedding.from_text(hyphal)

        print(f"embedding {hyphal} content: {content}")
        print(chunks)
        embeddings = HyphalEmbedding.embed_chunks(chunks)
        return HyphalEmbedding(content, embeddings)


    @staticmethod
    def from_text(path: Path):
        doc_loader = TextLoader(str(path))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        content = ""

        chunks = doc_loader.load_and_split(text_splitter=splitter)

        content = f"total of {len(chunks)} chunks"

        return content, chunks

    @staticmethod
    def from_pdf(path: Path):
        doc_loader = PyPDFLoader(str(path))
        content = ""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        chunks = doc_loader.load_and_split(text_splitter=splitter)

        content = f"total of {len(chunks)} chunks"

        return content, chunks

    @staticmethod
    def embed_chunks(chunks: list[str]):
        oembed = OllamaEmbeddings(
            base_url="http://localhost:11434", model="mixtral", show_progress=True
        )

        embedding = oembed.embed_documents(chunks)

        return embedding

    @staticmethod
    def embed_str(content, hyphal):
        print(f"working on embedding {content}")
        oembed = OllamaEmbeddings(
            base_url="http://localhost:11434", model="mixtral", show_progress=True
        )

        embedding = oembed.embed_documents([content])[0]

        return embedding

    @staticmethod
    def make_index(hyphal_document) -> tuple[str, AnnoyIndex, dict]:
        embeddings = [i.embedding for i in hyphal_documents]
        f = len(embeddings[0])
        t = AnnoyIndex(f, "angular")
        index_to_path = {}  # Mapping from index to file path
        for i, embedding in enumerate(embeddings):
            t.add_item(i, embedding)
            index_to_path[i] = str(tree[i])

        t.build(INDEX_TREES)
        annoy_index_path = source_path / ".hyphal" / "index.ann"
        t.save(str(annoy_index_path))
        return str(annoy_index_path), t, index_to_path


def make_hyphal(source_path):
    print(f"making hyphal at: {source_path}")

    source_path = Path(source_path)

    tree = helpers.get_tree()
    print(tree)

    hyphal_document = HyphalDocument.from_tree(source_path, tree)

    def write_hyphals():
        path_hyphal = source_path / ".hyphal"
        for i, filename in enumerate(tree):
            path = path_hyphal / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.with_suffix(".embedding")
            print(f"saving embedding to {path}")
            np.save(path, hyphal_document.embedding)
        return path_hyphal

    write_hyphals()

    path_index, index, index_to_path = make_index()

    return


def find_similar_documents(new_vector, index, index_to_path, n=10):
    similar_indices = index.get_nns_by_vector(new_vector, n, include_distances=True)
    distances = similar_indices[1]
    softmax_scores = softmax(-np.array(distances))

    results = []
    for idx, score in zip(similar_indices[0], softmax_scores):
        document_path = index_to_path[idx]
        embedding_path = Path(document_path).with_suffix(".embedding")
        embedding_value = np.load(embedding_path)
        results.append(
            {
                "document_path": document_path,
                "confidence": score,
                "embedding_path": str(embedding_path),
                "embedding": embedding_value,
            }
        )

    return results


def main():
    hyphal = cli.run(
        description="creates skeleton based on tree",
        func=make_hyphal,
        default_argument="./data",
        required=False,
        help_text="path to directory to generate mirrored file paths",
    )

    # query = "what should i do now"
    # embedded_query = embed_documents()

    print(hyphal)


if __name__ == "__main__":
    main()
