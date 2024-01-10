import os

from openai import OpenAI
from upstash_redis import Redis

class DependencyFactory:
    
    # Service Clients
    _openai_client = None
    _redis_client = None
    
    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), organization=os.environ.get("OPENAI_ORG"))
        return self._openai_client
    
    @property
    def redis_client(self):
        if self._redis_client is None:
            self._redis_client = Redis(url=os.environ.get("UPSTASH_REDIS_REST_URL"), token=os.environ.get("UPSTASH_REDIS_REST_TOKEN"))
        return self._redis_client
    
    
dependency_factory = DependencyFactory()