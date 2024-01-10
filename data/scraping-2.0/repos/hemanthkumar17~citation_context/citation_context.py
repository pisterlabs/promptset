
import pandas as pd
import fitz
import glob

from langchain.embeddings import HuggingFaceBgeEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore

base_path = "arxiv_mine/"

def embed_text(text, cache_name):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    fs = LocalFileStore("./cache/" + cache_name)

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        hf, fs, namespace=model_name
    )

    return cached_embedder.embed_documents(text)
def get_reference_vectors(row):
    print(row)
    ref_filename = row["label"]
    filename = row["file_name"]
    print(ref_filename)
    if ref_filename:
        directory = base_path + filename.split(".")[-2].replace("_", " ") + "/references"
        print(directory)
        file = None
        for x in glob.glob(directory + "/*.pdf"):
            if ref_filename.replace(" ", "_").lower() in x.lower():
                file = x
                break
        if not file:
            print("Reference file not found")
            return
        
        doc = fitz.open(file)
        blocks = []
        for page in doc:
            blocks += page.get_text("blocks")
        processed_blocks = [b[4].replace("\n", " ") for b in blocks if len(b[4].replace("\n", " ").split(" ")) > 20 and not (b[4][0] == "[" and (b[4][2] == "]" or b[4][3] == "]" or b[4][4] == "]"))]
        print("processed")

def format_data(data):
    citation_data = pd.read_parquet(data).explode("label", ignore_index=True).fillna("")
    # for data in citation_data:
    #     print(data)
    #     break
    print(citation_data["label"])
    citation_data.apply(get_reference_vectors, axis=1)

if __name__ == "__main__":
    format_data("citation_context.parquet")