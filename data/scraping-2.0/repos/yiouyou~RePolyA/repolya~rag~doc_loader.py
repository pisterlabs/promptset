from repolya._log import logger_rag

from langchain.document_loaders import PyMuPDFLoader

import os
import re


##### docs
def clean_txt(_txt):
    _txt = re.sub(r"\n+", "\n", _txt)
    _txt = re.sub(r"\t+", "\t", _txt)
    _txt = re.sub(r' +', ' ', _txt)
    _txt = re.sub(r'^\s+', '', _txt, flags=re.MULTILINE)
    return _txt

def get_docs_from_pdf(_fp):
    _f = os.path.basename(_fp)
    logger_rag.info(f"{_fp}")
    logger_rag.info(f"{_f}")
    loader = PyMuPDFLoader(str(_fp))
    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_txt(doc.page_content)
        # print(doc.metadata)
    logger_rag.info(f"load {len(docs)} pages")
    return docs

