from llmsherpa.readers import LayoutPDFReader
import cohere
import os
from dotenv import load_dotenv

load_dotenv('.env')

co = cohere.Client(os.getenv('COHERE_KEY'))
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"

"""parse the pdf contents and embed each item using cohere api"""


def embed_pdf(pdf_url):

    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    doc = pdf_reader.read_pdf(pdf_url)

    texts = [chunk.to_context_text() for chunk in doc.chunks()]

    doc_emb = co.embed(texts, input_type="search_document",
                       model="embed-english-v3.0").embeddings
    return {
        "doc": doc,
        "texts": texts,
        "embedded": doc_emb
    }


def embed_texts(texts):

    doc_emb = co.embed(texts, input_type="search_document",
                       model="embed-english-v3.0").embeddings
    return {
        "texts": texts,
        "embedded": doc_emb
    }
