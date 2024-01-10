from typing import List

import openai

from markdown_search.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


class OpenAIEncoder:
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        self.info = openai.Model.retrieve(self.model)

    def encode(self, text: str) -> list:
        return openai.Embedding.create(
            model=self.model,
            input=text
        ).data[0].embedding

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [
            record.embedding
            for record in
            openai.Embedding.create(
                model=self.model,
                input=texts
            ).data
        ]


if __name__ == '__main__':
    encoder = OpenAIEncoder()
    print(encoder.info)

    res = encoder.encode("hello world")
    print(len(res))
