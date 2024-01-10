
from InstructorEmbedding import INSTRUCTOR
from langchain.utils.math import cosine_similarity
import os
import re
import pickle

model = INSTRUCTOR('hkunlp/instructor-xl')
instruction = "Represent the Science title:"

def save_em(dirs, wpath):
    dirs1 = os.listdir(dirs)

    dct_dump = {}
    for dir in dirs1:
        path = os.path.join(dirs, dir)
        with open(path) as f:
            for line in f:
                list1 = line.replace("[", "").replace("]", "").replace("'", "").strip().split(",")
                item_all = []
                for item in list1:
                    item = item.replace(" ", "")
                    item_all.append([instruction, item])
            embeddings_item = model.encode(item_all)
            dct_dump[dir] = (embeddings_item, list1)
    with open(wpath, 'wb') as fp:
        pickle.dump(dct_dump, fp)

wpath = "files/data_b1.pickle"
dirs = "folder1"
save_em(dirs, wpath)
    
