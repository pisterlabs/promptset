from llms.llm_base import LLMBase
import openai
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings

# OpenAI w/ Langchain
class LLMOpenAI(LLMBase):

    def __init__(self, model:str, key:str):
        self.model = model
        openai.api_key = key
    
    # Chat directly using openai only
    def chat_completion(self, 
                        prior_messages: list[any],
                        neo4j_uri: str,
                        neo4j_user: str,
                        neo4j_password: str):
       full_response = ""
       for response in openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in prior_messages
            ],
            stream=True,
        ):
        full_response += response.choices[0].delta.get("content", "")
        return full_response