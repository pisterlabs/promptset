from llms.llm_base import LLMBase
import openai
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings

# OpenAI + Neo4j + Langchain
class LLMVectorSearch(LLMBase):

    def __init__(self, model:str, key:str):
        self.model = model
        openai.api_key = key

    def chat_completion(self, 
                        prior_messages: list[any],
                        neo4j_uri: str,
                        neo4j_user: str,
                        neo4j_password: str):

        # latest prompt = last prior_message
        query = prior_messages[-1]['content']

        documents = [d['content'] for d in prior_messages[:-1]]
        
        # TODO: Untested
        neo4j_vector = Neo4jVector.from_documents(
            documents,
            OpenAIEmbeddings(),
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )

        results = neo4j_vector.similarity_search(query, k=1)
        return results[0].page_content