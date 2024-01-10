# pdf loader
from langchain.document_loaders import PyPDFLoader, ArxivLoader
# url loader
from langchain.document_loaders import UnstructuredURLLoader
# youtube loader
from langchain.document_loaders import YoutubeLoader

# time module
from tqdm import tqdm

import re


from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def load_paper(config):
    """Load paper from pdf or arxiv

    Args:
        config (dict): argument parser
    """    
    ## PDF Loader
    if config.pdf:
        loader = PyPDFLoader(config.pdf)
        load_paper = loader.load()
        assert load_paper, "Invalid pdf file"
        config.paper = load_paper
        print(f"page number: {len(load_paper)}")

    ## Arxiv Loader
    if config.arxiv:
        loader = ArxivLoader(query=config.arxiv)
        load_paper = loader.load()
        assert load_paper, "Invalid arxiv number"
        config.paper = load_paper
        print(f"page number: {len(load_paper)}")
    
    ## Youtube Script Loader
    if config.html:
        urls = [config.html]
        loader = UnstructuredURLLoader(urls=urls)
        load_paper = loader.load()

        assert load_paper, "Invalid html file"
        config.paper = load_paper
        print(f"page number: {len(load_paper)}")

    if config.youtube:
        loader = YoutubeLoader.from_youtube_url(youtube_url=config.youtube, add_video_info=True)
        load_paper = loader.load()
        assert load_paper, "Invalid html file"
        config.paper = load_paper
        print(f"page number: {len(load_paper)}")


