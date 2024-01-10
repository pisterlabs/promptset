import os

from openai import OpenAI

class DependencyFactory:
    
    # Service Clients
    _openai_client = None
    
    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), organization=os.environ.get("OPENAI_ORG"))
        return self._openai_client
    
dependency_factory = DependencyFactory()