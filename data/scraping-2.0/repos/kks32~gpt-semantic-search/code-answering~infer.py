"""This is the logic for ingesting data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
from glob import glob
import pandas as pd
import numpy as np

# Root directory where the Python repo is located
ROOT_DIR = "/Users/krishna/research/mpm/LearnMPM/"
CODE_REPO = "LearnMPM"

import os
from glob import glob
import pandas as pd
import numpy as np

def get_function_name(code):
    """
    Extract function name from a line beginning with "def "
    """
    assert code.startswith("def ")
    return code[len("def "): code.index("(")]

def get_until_no_space(all_lines, i) -> str:
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, i + 10000):
        if j < len(all_lines):
            if len(all_lines[j]) == 0 or all_lines[j][0] in [" ", "\t", ")"]:
                ret.append(all_lines[j])
            else:
                break
    return "\n".join(ret)

def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    whole_code = open(filepath).read().replace("\r", "\n")
    all_lines = whole_code.split("\n")
    for i, l in enumerate(all_lines):
        if l.startswith("def "):
            code = get_until_no_space(all_lines, i)
            function_name = get_function_name(code)
            yield (filepath + ":" + function_name, code)


# get user root directory
root_dir = os.path.expanduser(ROOT_DIR)

# path to code repository directory
code_root = root_dir + CODE_REPO

code_files = [y for x in os.walk(code_root) for y in glob(os.path.join(x[0], '*.py'))]
print("Total number of py files:", len(code_files))

if len(code_files) == 0:
    print("Double check that you have downloaded the repo and set the code_root variable correctly.")

docs = []
metadatas = []
for code_file in code_files:
    funcs = list(get_functions(code_file))
    for func in funcs:
        docs.append(func[0])
        metadatas.extend([{"source": func[1]}])

print("Total number of functions extracted:", len(docs))


# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
