import os

import diskcache
import platformdirs

from guidance.llms.caches import Cache

dict = {}
class DiskCache(Cache):
    """DiskCache is a cache that uses diskcache lib."""
    def __init__(self, llm_name: str):
        pass

    def __getitem__(self, key: str) -> str:
        return dict[key]

    def __setitem__(self, key: str, value: str) -> None:
        dict[key] = value

    def __contains__(self, key: str) -> bool:
        return False
    
    def clear(self):
        pass
