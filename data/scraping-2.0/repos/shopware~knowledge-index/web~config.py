import os
from typing import Union
from pydantic import BaseSettings
from .utils import safe_dir_append


def prefix(path: str) -> str:
    root = os.environ.get("ROOT_DIR", "/")
    return os.path.join(root, path)


def get_embedding_fn():
    if "OPENAI_API_KEY" in os.environ:
        from langchain.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
    else:
        from langchain.embeddings import TensorflowHubEmbeddings

        url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        return TensorflowHubEmbeddings(model_url=url)


def env_dir(env, dir, collection: str = None):
    if collection:
        env = safe_dir_append(env, "_" + collection.upper())
        dir = safe_dir_append(dir, "-" + collection)

    return {"env": env, "dir": dir}


def data_dir(collection: str = None):
    conf = env_dir("DATA_DIR", prefix("data/docs"), collection)

    return os.environ.get(conf["env"], conf["dir"])


def db_dir(collection: str = None):
    conf = env_dir("DB_DIR", prefix("data/db"), collection)

    return os.environ.get(conf["env"], conf["dir"])


def cache_dir():
    conf = env_dir("CACHE_DIR", prefix("data/cache"), None)

    return os.environ.get(conf["env"], conf["dir"])


def sqlite_dir(collection: str = None):
    conf = env_dir("SQLITE_DIR", prefix("data/sqlite"), collection)

    return os.environ.get(conf["env"], conf["dir"])


# not used yet
class Settings(BaseSettings):
    # api keys
    openai_api_key: Union[str, None] = None
    knowledge_api_key: Union[str, None] = None
    # data dirs
    root_dir: str = '/'
    data_dir: str = '/data/docs'
    db_dir: str = '/data/db'
    cache_dir: str = '/data/cache'
    sqlite_dir: str = '/data/sqlite'


settings = Settings()
