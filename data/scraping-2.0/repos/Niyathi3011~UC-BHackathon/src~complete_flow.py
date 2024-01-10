from pathlib import Path
from arxiv_engine.utils import ArxivWrapper, NodeManager
from arxiv_engine.server_config import BASE_DATABASE_PATH
from llama_index import download_loader
from arxiv_engine.chroma_api import ChromaIndex, ChromaRetriever

import openai
openai.api_key = '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'
import os
os.environ["OPENAI_API_KEY"] = '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'

PDFReader = download_loader("PDFReader")
loader = PDFReader()
node_manager = NodeManager()

arxiv_wrapper = ArxivWrapper()
title_categories_dict = arxiv_wrapper.download_datasets()
print(title_categories_dict)

def load_document_in_llama(title_categories_dict):
    categories_text_dict = {}
    # documents_list = []
    for title, categories in title_categories_dict.items():
        print(title, categories)
        file_path = BASE_DATABASE_PATH + categories + "/" + title + ".pdf"
        print(file_path)
        documents = loader.load_data(file=Path(file_path))
        print(documents)
        # documents_list.append(documents)
        categories_text_dict[categories] = documents
    return categories_text_dict

def get_text_data(categories_text_dict):
    document_text_dict = {}
    for category, documents in categories_text_dict.items():
        documents_text = ""
        for pages in documents:
            documents_text = documents_text + pages.get_text() + " "
        title = pages.extra_info['file_name']
        document_text_dict[title] = {
            'document_text': documents_text,
            'category': category
        }
    return document_text_dict

categories_text_dict = load_document_in_llama(title_categories_dict)
document_text_dict = get_text_data(categories_text_dict)

print(document_text_dict)

node_list = node_manager.construct_node(document_text_dict)
chroma_index = ChromaIndex(node_list)
chroma_index.retreiver('Tell me more about the paper AI')
# query_engine = chroma_index.as_query_engine(
#     chroma_collection=chroma_collection
# )
# response = query_engine.query("What did the author do growing up?")
# chroma_retriever = ChromaRetriever(chroma_index)
# output = chroma_retriever.retriever.retrieve('Tell me more about the paper AI')
# print(output)
