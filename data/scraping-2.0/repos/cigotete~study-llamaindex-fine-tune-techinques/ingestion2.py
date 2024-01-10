from pathlib import Path
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.llms import OpenAI
from llama_index.node_parser.text import SentenceSplitter
from llama_index.vector_stores import PineconeVectorStore
import pinecone

# Based on https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval.html

if __name__ == "__main__":

    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=Path('./docs/document.pdf'))


    text_parser = SentenceSplitter(
        chunk_size=1024,
        # separator=" ",
    )

    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    from llama_index.schema import TextNode

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

