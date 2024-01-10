import langchain

from langchain.llms import OpenAI
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain

# text-davinci-002 速度较慢
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

# ############### InMemoryCache ################

def InMemoryCacheDemo():
    langchain.llm_cache = InMemoryCache()
    print(llm.predict("Tell me a joke"))
    print(llm.predict("Tell me a joke"))

# ############### SQLite Cache ################

def SQLiteCacheDemo():
    langchain.llm_cache = SQLiteCache(database_path="./files/.langchain.db")
    print(llm.predict("Tell me a joke"))
    print(llm.predict("Tell me a joke"))


if __name__ == "__main__":
    # InMemoryCacheDemo()
    SQLiteCacheDemo()
    