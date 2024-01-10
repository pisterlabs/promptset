import json
import os
import time

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_TYPE"] = ""
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""

embeddings = OpenAIEmbeddings(model='embeddingv2', chunk_size=1)

text_embeddings = []
metadatas = []

start = time.perf_counter()  # FIXME

with open('/Users/crown/Downloads/blocks.txt.txt', 'r+') as f:
    lines = f.readlines()
    for line in lines:
        # print(line)
        text_embeddings.append((
            'xxx',
            json.loads(line)['sentence_vector'],
        ))
        metadatas.append({})

print(len(text_embeddings))

runTime_load = time.perf_counter() - start
print("文件 load 时间：", runTime_load * 1000, "毫秒")

ix = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embeddings, metadatas=metadatas)
ix.save_local('/Users/crown/Projects/python101/python-playground')

runTime_savelocal = time.perf_counter() - start
print("文件 save 时间：", runTime_savelocal * 1000, "毫秒")

ix = FAISS.load_local(folder_path='/Users/crown/Projects/python101/python-playground', embeddings=embeddings)

runTime_loadlocal = time.perf_counter() - start
print("文件 save 时间：", runTime_loadlocal * 1000, "毫秒")

if __name__ == '__main__':
    pass
