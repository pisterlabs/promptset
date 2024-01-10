
from InstructorEmbedding import INSTRUCTOR
from langchain.utils.math import cosine_similarity
import os
import re
import pickle

model = INSTRUCTOR('hkunlp/instructor-xl')
instruction = "Represent the Science title:"

def get_relevant_tables(query, wpath):
    embeddings_query = model.encode([[instruction,query]])

    file = open(wpath, "rb")
    dct_dump = pickle.load(file)

    dct = {}
    key_item = set()
    for dir in dct_dump:
        embeddings_item, item_q = dct_dump[dir]
        score = cosine_similarity(embeddings_item, embeddings_query)
        score_all = [num[0] for num in score]
        score_all.sort(reverse=True)
        flag = False
        for item in item_q:
            item = item.replace(" ", "")
            if item in query.split(" "):
                flag = True
                key_item.add(item)
        if flag:
            score_all[0] = 1.0
        sum =0.0
        for score_ in score_all[:5]:
            sum += score_
        average = sum / 5
        dct[dir] = average
    dct_order = sorted(dct.items(), key=lambda x:x[1], reverse=True)
    return dct_order, key_item

query = "what is the average totalamountlocal for last three years? startdate"
wpath = "files/data_b1.pickle"
dct_order, key_item = get_relevant_tables(query, wpath)

def get_schema():
    result = []
    _ = 0
    for order in dct_order:
        _ += 1
        if _ > 1:
            break
        for key in key_item:
            result.append(order[0].replace(".txt", "") + "." + key)
    return str(result).replace(" ", "")

print(get_schema())
