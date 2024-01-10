import os
from langchain.schema import Document
from prompt_stats.get_stats import get_doc_length
from langchain.document_loaders import TextLoader


def get_dita_docs(min_doc_length=0) -> list[Document]:
    dita_dict = {}
    dita_path = 'data/topics/'

    def func(lst, p):
        for fn in os.listdir(p):
            if os.path.isfile(p + fn) and fn[-5:] == '.dita':
                dita_dict[p + fn] = get_doc_length(p + fn)
            elif os.path.isdir(p + fn):
                func(lst, p + fn + '/')

    func(dita_dict, dita_path)
    path_list_less = [i for i, j in dita_dict.items() if j > min_doc_length]
    path_list = list(dita_dict.keys())

    docs = TextLoader(path_list[0], encoding='utf-8').load()
    split_docs = [TextLoader(pt, encoding='utf-8').load() for pt in path_list_less[1:]]
    for i in split_docs:
        docs += i

    return docs
