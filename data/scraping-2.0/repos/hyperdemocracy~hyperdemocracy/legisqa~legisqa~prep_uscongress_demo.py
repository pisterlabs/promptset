"""

"""

from pathlib import Path

from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
import pandas as pd

from hyperdemocracy import langchain_helpers
from hyperdemocracy.datasets import uscongress


def write_lc_docs_from_hf_ds(out_file):
    hf_org = "hyperdemocracy"
    hf_dataset_name = "us-congress-bills"
    ds = load_dataset(hf_org + "/" + hf_dataset_name)

    docs = uscongress.get_langchain_docs(ds['train'])
    langchain_helpers.write_docs_to_jsonl(docs, out_file)


def write_split_docs(docs, chunk_size, chunk_overlap, out_file):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        add_start_index = True,
    )
    split_docs = text_splitter.split_documents(docs)
    langchain_helpers.write_docs_to_jsonl(split_docs, out_file)


docs_file = "./lc_docs.jsonl"
split_docs_file = "./lc_split_docs.jsonl"
chunk_size = 1024
chunk_overlap = 256


write_lc_docs_from_hf_ds(docs_file)
docs = langchain_helpers.read_docs_from_jsonl(docs_file)

write_split_docs(docs, chunk_size, chunk_overlap, split_docs_file)
split_docs = langchain_helpers.read_docs_from_jsonl(split_docs_file)


model_tag = "bge-small-en"
model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedder = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


vecs = embedder.embed_documents([doc.page_content for doc in split_docs])
rows = []
for doc, vec in zip(split_docs, vecs):
    metadata = doc.metadata.copy()
    doc_id = metadata.pop("id")
    chunk_id = "{}-{}".format(doc_id, metadata["start_index"])

    metadata["chunk_id"] = chunk_id
    metadata["parent_id"] = doc_id
    row = {
        "id": chunk_id,
        "text": doc.page_content,
        "metadata": metadata,
        "vec": vec,
    }
    rows.append(row)

df = pd.DataFrame(rows)
outfile = f"uscb.c118.s{chunk_size}.o{chunk_overlap}.{model_tag}.parquet"
df.to_parquet(outfile)
