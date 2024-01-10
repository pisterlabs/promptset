import os
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.tools import Tool
from twk_backend.tools.datastore_query.utils import get_customer_support_template


def get_datastore_query_tool(
    datastore_id: str,
    datastore_name: str,
    datastore_description: str = "It is a datasource",
):
    datastore_query = DatastoreQuery(datastore_id=datastore_id)
    tool = Tool(
        name=datastore_name,
        description=f"""
        Useful for answering questions about: {datastore_name}.
        {datastore_description}.
        Input must be a fully formed question.
        """,
        func=datastore_query.chat,
    )
    tool.return_direct = True

    return tool


class DatastoreQuery:
    top_k: int = 3

    def __init__(self, datastore_id: str) -> None:
        self.datastore_id = datastore_id
        self.headers = {
            "api-key": os.getenv("QDRANT_API_KEY"),
        }
        self.base_url = os.getenv("QDRANT_API_URL")
        self.embeddings = OpenAIEmbeddings()
        self.model = OpenAI()

    def process_results(self, results):
        result_data = results.get("result", [])
        return [
            {
                "score": each.get("score"),
                "source": each.get("payload", {}).get("source"),
                "text": each.get("payload", {}).get("text"),
            }
            for each in result_data
        ]

    def search(self, query: str):
        vectors = self.embeddings.embed_documents([query])

        response: requests.Response = requests.post(
            url=f"{self.base_url}/collections/text-embedding-ada-002/points/search",
            headers=self.headers,
            json={
                "vector": vectors[0],
                "limit": self.top_k,
                "with_payload": True,
                "with_vectors": False,
                "filter": {
                    "must": [
                        {"key": "datastore_id", "match": {"value": self.datastore_id}}
                    ]
                },
            },
        )

        results = response.json()

        return self.process_results(results=results)

    def chat(self, query: str):
        results = self.search(query)
        context = "\n\n".join(
            f"CHUNK: {each.get('text')}\nSOURCE: {each.get('source')}"
            for each in results
        )

        final_prompt = get_customer_support_template(query=query, context=context)

        prompt_template = PromptTemplate.from_template(final_prompt)

        chain = LLMChain(llm=self.model, prompt=prompt_template)

        output = chain.run({})

        return output.strip()
