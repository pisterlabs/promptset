from gptcache import cache
from gptcache.embedding import OpenAI
from gptcache.manager import VectorBase, get_data_manager, CacheBase
from gptcache.similarity_evaluation import SearchDistanceEvaluation
from src.codevecdb.config.Config import Config


def get_content_func(data, **_):
    return data.get("prompt").split("Question")[-1]


def cache_initialize():
    print("start init cache")
    openai = OpenAI()
    cache_base = CacheBase('sqlite')

    cfg = Config()
    if cfg.milvus_secure == "True" or cfg.milvus_secure == "true":
        secure = True
    else:
        secure = False
    vector_base = VectorBase('milvus',
                             host=cfg.milvus_host,
                             port=cfg.milvus_port,
                             user=cfg.milvus_user,
                             secure=secure,
                             password=cfg.milvus_password,
                             dimension=openai.dimension,
                             collection_name="milvus_cache")

    data_manager = get_data_manager(cache_base, vector_base)

    cache.init(
        pre_embedding_func=get_content_func,
        embedding_func=openai.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(max_distance=0.5, positive=True))

    cache.set_openai_key()
    print("success init cache")
