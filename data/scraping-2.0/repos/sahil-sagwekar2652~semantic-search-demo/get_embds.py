from langchain.embeddings.base import Embeddings
from typing import List
import requests


class LocalLlamaEmbeddings(Embeddings):
    def __init__(self, url: str, headers: dict):
        self.url = url
        self.headers = headers
        super().__init__()

    def api_query(self, url: str, headers: dict, json: dict):
        response = requests.post(url=self.url, headers=self.headers, json=json)
        response = response.json()
        return response["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # embeddings = []

	    async def post_multiple():
		    async with aiohttp.ClientSession() as session:
		    	tasks = []
		        for text in texts:  # replace with your range
		            url = self.url  # replace with your API endpoint
		            payload = {"input": text}  # replace with your payload
		            async with session.post(url, json=payload) as resp:
		                data = await resp.json()
		                tasks.append(data)
		        responses = await asyncio.gather(*tasks)

		embeddings = asyncio.run(post_multiple())


        # for text in texts:
        #     data = {
        #         "input": text,
        #     }
        #     embeddings.append(
        #         self.api_query(url=self.url, headers=self.headers, json=data)
        #     )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.api_query(url=self.url, headers=self.headers, json={"input": text})




async def post_multiple():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for number in range(1, 151):  # replace with your range
            url = f'https://your-api-endpoint/{number}'  # replace with your API endpoint
            payload = {"key": "value"}  # replace with your payload
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                tasks.append(data)
        responses = await asyncio.gather(*tasks)
        # process responses here
