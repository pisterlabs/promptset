from src.chroma.chroma_utils import load_document, split_document
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

sources = [
    "https://jazzobserver.com/the-origins-of-jazz/",
    "data/test_documents/the_origins_of_jazz.md",
    "data/test_documents/the_origins_of_jazz.txt",
    "data/test_documents/jazz.csv",
    "data/test_documents/influence of jazz on blues.pdf",
    "https://gladsaxe.dk/kommunen/borger/borgerservice/pas-koerekort-og-id/pas",
]

# test loader
docs = []
for source in sources:
    print(f"source: {source}")
    metadata = load_document(source)[0].metadata
    assert metadata["source"] == source
    docs.append(load_document(source))
    print(metadata)
    print("----------------------------------------------------")

# test splitter
for doc in docs:
    print(f"doc: {doc[0].metadata['source']}")
    chunks = split_document(document=doc)
    print(f"number of chunks: {len(chunks)}")
    assert len(chunks) > 0
    print("----------------------------------------------------")
