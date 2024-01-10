import os
import pandas as pd
from shortuuid import uuid

from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer


def load_docs(input_path: str, output_path: str = None) -> pd.DataFrame:
    """Load documents from input_path and return a DataFrame containing doc_id, doc_name and doc_text"""
    docs = pd.DataFrame(columns=["doc_id", "doc_name", "doc_text"])
    for i, doc_name in enumerate(os.listdir(input_path)):
        with open(f"{input_path}/{doc_name}", "r") as f:
            docs.loc[i] = {"doc_id": uuid(), "doc_name": doc_name, "doc_text": f.read()}

    save_output(docs, filename="docs.csv", output_path=output_path)
    return docs


def get_chunks(docs: pd.DataFrame, output_path: str = None, **kwargs) -> pd.DataFrame:
    """Split documents into chunks and return a DataFrame containing doc_id, chunk_id and chunk_text"""
    text_splitter = CharacterTextSplitter(**kwargs)
    chunks = pd.DataFrame(columns=["doc_id", "chunk_id", "chunk_text"])
    for _, doc in docs.iterrows():
        for chunk in text_splitter.create_documents([doc.doc_text]):
            chunks.loc[len(chunks)] = {
                "doc_id": doc.doc_id,
                "chunk_id": uuid(),
                "chunk_text": chunk.page_content,
            }

    save_output(chunks, filename="chunks.csv", output_path=output_path)
    return chunks


def get_embeddings(chunks: dict, output_path: str = None, **kwargs) -> pd.DataFrame:
    """Get embeddings for chunks and return a DataFrame containing doc_id, chunk_id and embeddings"""
    embeddings_model = SentenceTransformer(**kwargs)
    embeddings = pd.DataFrame(columns=["doc_id", "chunk_id", "embeddings"])
    for _, chunk in chunks.iterrows():
        embeddings.loc[len(embeddings)] = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "embeddings": embeddings_model.encode(chunk.chunk_text),
        }
    # Expand embeddings into columns
    embeddings_expanded = pd.DataFrame(embeddings.pop("embeddings").to_list())
    embeddings = pd.concat([embeddings, embeddings_expanded], axis=1)

    save_output(embeddings, filename="embeddings.csv", output_path=output_path)
    return embeddings


def save_output(df: pd.DataFrame, filename: str, output_path: str = None) -> None:
    """Save DataFrame to output_path"""
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(f"{output_path}/{filename}", index=False)
