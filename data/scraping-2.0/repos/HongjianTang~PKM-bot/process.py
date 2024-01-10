from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import sys
from replit import db
import json


def train():
  if len(db.keys()) < 1:
    print("The database should contain at least one key-value pair.",
          file=sys.stderr)
    return

  data = []
  for key in db.keys():
    print(f"Add data from key {key} to dataset")
    data.append(json.loads(db[key]))

  # 文本切块，后续需考虑优化md和代码场景
  textSplitter = CharacterTextSplitter(chunk_size=2000, separator="\n")

  docs = []
  for sets in data:
    ttt = sets["date"] + " " + " " + sets["text"]
    docs.extend(textSplitter.split_text(ttt))

  # 调用ada模型向量化文本
  print(docs)
  store = FAISS.from_texts(docs, OpenAIEmbeddings())
  faiss.write_index(store.index, "training.index")
  store.index = None

  with open("training/faiss.pkl", "wb") as f:
    pickle.dump(store, f)
