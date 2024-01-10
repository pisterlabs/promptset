import json
from typing import List
from qdrant_client import QdrantClient
from openai import OpenAI

from rag.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_RAG_COLLECTION_NAME



class Searcher:

    def __init__(self):
        self.collection_name = QDRANT_RAG_COLLECTION_NAME
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    def search(self, question, limit=5) -> List[dict]:
        contextArray = self.client.query(
                collection_name=self.collection_name,
                query_text=question,
                limit=limit,
            )
        return contextArray

class OpenAICaller:
        def __init__(self):
            self.client = OpenAI()
        
        def query(self,metaprompt):
            response = self.client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": metaprompt},
            ],
            timeout=10.0)
            return response.choices[0].message.content

class LLMQuery:
    def __init__(self):
        self.searcher = Searcher()
        self.OpenAIclient = OpenAICaller()

    def query(self, question, limit=5) -> List[dict]:
        steps=[]
        steps.append({
            "name": "User Query",
            "context":f"""User Query: {question.strip()}"""
        })

        userQueryMetaprompt = f"""
            Given a piece of text containing personal information, generate a question or query related to Qdrant that removes any identifiable personal information. Ensure the output focuses solely on Qdrant-related content and is suitable for querying a vector database.

            User query: {question.strip()}
            """
        
        steps.append({
            "name": "User query to query text",
            "context": f"""
            User query is sent to OpenAI to convert to query text for vector database give query text for vector database.

            metaprompt: {userQueryMetaprompt.strip()}
            """
        })

        query_text = self.OpenAIclient.query(metaprompt=userQueryMetaprompt)
        steps.append({
            "name": "OpenAI response",
            "context": f""" OpenAI response: {query_text.strip()}"""
        })
      
        steps.append({
            "name": "Query text to db",
            "context": f"""
            Query text is sent to Qdrant database.
            
            Query text: {query_text.strip()}
            """
        })
        contextArray = self.searcher.search(query_text, limit=limit)
        context = "\n".join(r.document for r in contextArray)
        steps.append({
            "name": "DB response",
            "context":f""" DB returns the most similar points:""",
            "contextArray": contextArray,
        })

        dbResponseMetaprompt =f"""
            Answer the following question using the provided context.
            If you can't find the answer, do not pretend you know it, but answer "I don't know".

            Question: {query_text.strip()}

            Context:
            {context.strip()}

            Answer:
            """
        steps.append({
            "name": "DB response to answer",
            "context": f"""
            DB response is sent to OpenAI
            
            metaprompt: {dbResponseMetaprompt.strip()}

            """
        })

        answer = self.OpenAIclient.query(metaprompt=dbResponseMetaprompt)
        steps.append({
            "name": "Answer",
            "context": answer
        })

        return({
            "response" :answer,
            "steps":steps
        })

        


if __name__ == '__main__':
    question = "I am a student from India and looking for AI projects to contribute, I like qdrant but I am having problem with timeout error."

    searcher = LLMQuery()

    res = searcher.query(question)
    print(res)