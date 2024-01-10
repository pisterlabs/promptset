"""Contains confluence loaders for loading data from confluence into the vectorstore."""
import logging
import sys

from langchain.schema.vectorstore import VectorStore
from lxml import etree
from pulse.services.confluence import confluencefactory
from pulse import config

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger("pulse.loaders.confluence")

def load_all_labelled_docs(vectorstore: VectorStore):
    """
    Load all documents from a confluence space with "pulse" label into the vector store
    """
    # rest call to confluence with the username and api key
    docs = confluencefactory.client.get_all_pages_from_space(config.confluence_space, expand="body.view,version")
    for doc in docs:
        doc_loader(doc, vectorstore)
        
        
def doc_loader(doc: dict, vectorstore: VectorStore):
    """
    Load confluence documents into the vectorstore.
    """
    # rest call to confluence with the username and api key
    text = _strip_html_tags(doc.get("body").get("view").get("value"))
    
    meta = {k: v for k, v in doc.items() if k != "body"}
    meta["source"] = "confluence"
    
    logger.debug("Adding document to vectorstore: %s, {%s}", text, meta)
    
    vectorstore.aadd_texts(texts=[text], metadata=meta)
    logging.debug("Added document to vectorstore: %s", id)
    
def _strip_html_tags(text: str) -> etree._ElementUnicodeResult:
    """Remove html tags from a string"""
    return etree.HTML(text).xpath("string()")
