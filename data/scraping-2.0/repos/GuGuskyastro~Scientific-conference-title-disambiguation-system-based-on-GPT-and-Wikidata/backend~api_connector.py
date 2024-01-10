import os
import weaviate
from langchain.chat_models import ChatOpenAI

# Initialize the connection with LLM and Weaviate VS

class APIConnector:
    def __init__(self,model_name="gpt-3.5-turbo"):
        # OpenAI API key
        os.environ['OPENAI_API_KEY']

        # Weaviate client
        self.client = weaviate.Client(
            url=os.environ['WEAVIATE_URL'],
            auth_client_secret=weaviate.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY']),
            additional_headers={
                "X-HuggingFace-Api-Key": os.environ['HUGGINGFACE_API_KEY']
            }
        )

        # OpenAI LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
