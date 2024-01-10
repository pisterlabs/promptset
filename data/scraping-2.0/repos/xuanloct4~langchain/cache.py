import environment

from langchain.llms import OpenAI

###In Memory Cache
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
# To make the caching really obvious, lets use a slower model.
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
# The second time it is, so it goes faster
llm("Tell me a joke")


###SQLite Cache
#!rm .langchain.db
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")

# The second time it is, so it goes faster
llm("Tell me a joke")



###Redis Cache
##Standard Cache
#Use Redis to cache prompts and responses.
#
# We can do the same thing with a Redis cache
# (make sure your local Redis instance is running first before running this example)
from redis import Redis
from langchain.cache import RedisCache

langchain.llm_cache = RedisCache(redis_=Redis())
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
# The second time it is, so it goes faster
llm("Tell me a joke")


##Semantic Cache
#Use Redis to cache prompts and responses and evaluate hits based on semantic similarity.
#
from langchain.embeddings import OpenAIEmbeddings
from langchain.cache import RedisSemanticCache

langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings()
)
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
# The second time, while not a direct hit, the question is semantically similar to the original question,
# so it uses the cached result!
llm("Tell me one joke")



###GPTCache
##Let’s first start with an example of exact match
import gptcache
from gptcache.processor.pre import get_prompt
from gptcache.manager.factory import get_data_manager
from langchain.cache import GPTCache

# Avoid multiple caches using the same file, causing different llm model caches to affect each other
i = 0
file_prefix = "data_map"

def init_gptcache_map(cache_obj: gptcache.Cache):
    global i
    cache_path = f'{file_prefix}_{i}.txt'
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=get_data_manager(data_path=cache_path),
    )
    i += 1

langchain.llm_cache = GPTCache(init_gptcache_map)
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
# The second time it is, so it goes faster
llm("Tell me a joke")




##Let’s now show an example of similarity caching
import gptcache
from gptcache.processor.pre import get_prompt
from gptcache.manager.factory import get_data_manager
from langchain.cache import GPTCache
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache import Cache
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# Avoid multiple caches using the same file, causing different llm model caches to affect each other
i = 0
file_prefix = "data_map"
llm_cache = Cache()


def init_gptcache_map(cache_obj: gptcache.Cache):
    global i
    cache_path = f'{file_prefix}_{i}.txt'
    onnx = Onnx()
    cache_base = CacheBase('sqlite')
    vector_base = VectorBase('faiss', dimension=onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base, max_size=10, clean_size=2)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
    )
    i += 1

langchain.llm_cache = GPTCache(init_gptcache_map)
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
# This is an exact match, so it finds it in the cache
llm("Tell me a joke")
# This is not an exact match, but semantically within distance so it hits!
llm("Tell me joke")





###SQLAlchemy Cache
# You can use SQLAlchemyCache to cache with any SQL database supported by SQLAlchemy.

# from langchain.cache import SQLAlchemyCache
# from sqlalchemy import create_engine

# engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
# langchain.llm_cache = SQLAlchemyCache(engine)

###Custom SQLAlchemy Schemas
# You can define your own declarative SQLAlchemyCache child class to customize the schema used for caching. For example, to support high-speed fulltext prompt indexing with Postgres, use:

from sqlalchemy import Column, Integer, String, Computed, Index, Sequence
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import TSVectorType
from langchain.cache import SQLAlchemyCache

Base = declarative_base()


class FulltextLLMCache(Base):  # type: ignore
    """Postgres table for fulltext-indexed LLM Cache"""

    __tablename__ = "llm_cache_fulltext"
    id = Column(Integer, Sequence('cache_id'), primary_key=True)
    prompt = Column(String, nullable=False)
    llm = Column(String, nullable=False)
    idx = Column(Integer)
    response = Column(String)
    prompt_tsv = Column(TSVectorType(), Computed("to_tsvector('english', llm || ' ' || prompt)", persisted=True))
    __table_args__ = (
        Index("idx_fulltext_prompt_tsv", prompt_tsv, postgresql_using="gin"),
    )

engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
langchain.llm_cache = SQLAlchemyCache(engine, FulltextLLMCache)




###Optional Caching
#You can also turn off caching for specific LLMs should you choose. In the example below, even though global caching is enabled, we turn it off for a specific LLM

llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2, cache=False)
llm("Tell me a joke")
llm("Tell me a joke")





###Optional Caching in Chains
#You can also turn off caching for particular nodes in chains. Note that because of certain interfaces, its often easier to construct the chain first, and then edit the LLM afterwards.
#
#As an example, we will load a summarizer map-reduce chain. We will cache results for the map-step, but then not freeze it for the combine step.

llm = OpenAI(model_name="text-davinci-002")
no_cache_llm = OpenAI(model_name="text-davinci-002", cache=False)
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain

text_splitter = CharacterTextSplitter()
with open('../../../state_of_the_union.txt') as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)
from langchain.docstore.document import Document
docs = [Document(page_content=t) for t in texts[:3]]
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(llm, chain_type="map_reduce", reduce_llm=no_cache_llm)

chain.run(docs)
chain.run(docs)