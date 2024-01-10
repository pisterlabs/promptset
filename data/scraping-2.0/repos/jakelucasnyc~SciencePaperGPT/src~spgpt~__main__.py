from spgpt.pdf import retrieve_pdf_data, clear_cached_papers
from spgpt.query import get_response_from_query
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import logging
from appdirs import user_cache_dir, user_data_dir
import json
from pprint import pprint
from spgpt.gui.spgptMainWin import SPGPTMainWin
from PySide6.QtWidgets import QApplication
import sys

logging.basicConfig(level=logging.INFO)

cache_dir = user_cache_dir('SciencePaperGPT', 'BioGear Labs')

def create_paper_cache_file(cache_dir:str):
    cache_file = os.path.join(cache_dir, 'papers.json')
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.isfile(cache_file):
        with open(cache_file, 'w') as f:
            json.dump({}, f)
    return cache_file

if __name__ == '__main__':
    with open(os.path.realpath(os.path.join('spgpt', 'secrets', 'openai.txt'))) as f:
        os.environ['OPENAI_API_KEY'] = f.read()

    # clear_cached_papers(cache_dir)
    create_paper_cache_file(cache_dir)

    embeddings = OpenAIEmbeddings()
    # pdf = r'/Users/jakelucas/Desktop/NicotineTest.pdf'
    # pdf = r'/Users/jakelucas/Desktop/Niemeyer2022.pdf'
    # faiss_db = retrieve_pdf_data(pdf, embeddings, cache_dir)
    # response, docs = get_response_from_query(faiss_db, "What were the major results in this paper? ")
    # pprint(response)

    app = QApplication(sys.argv)
    main_window = SPGPTMainWin(embeddings, cache_dir)
    main_window.show()
    sys.exit(app.exec())


