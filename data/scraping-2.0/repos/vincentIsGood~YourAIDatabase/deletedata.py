import argparse
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import configs.common as config
from configs.common import model_config
from lib.utils.InteractiveConsole import InteractiveConsole, SimpleCommandHandler

cmdParser = argparse.ArgumentParser()
cmdParser.add_argument("-id", "--docids", action='append', default=[], help="Delete docs by ids")
cmdParsed = cmdParser.parse_args()

print("[+] Preparing Chroma DB")
embedding_func = SentenceTransformerEmbeddings(model_name=model_config.SENTENCE_EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
chromadb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embedding_func)

print("[+] Chroma index:")
print(chromadb._collection, "\n")

if len(cmdParsed.docids) > 0:
    print("[+] Deleting doc with ids: ", cmdParsed.docids)
    chromadb._collection.delete(cmdParsed.docids)