import os
from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


class EmbedOpenAI:
    def __init__(self, batch_size: Optional[int] = 1000) -> None:
        self.model = os.environ.get("EMBED_MODEL")
        self.batch_size = batch_size
        self._client = OpenAI(max_retries=5, timeout=30.0)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _get_embeddings(self, batch_text: List[str]) -> List[List[float]]:
        # replace newlines, which can negatively affect performance
        batch_text = [text.replace("\n", " ") for text in batch_text]
        data = self._client.embeddings.create(input=batch_text, model=self.model).data
        return [d.embedding for d in data]

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            embeddings.extend(self._get_embeddings(texts[i : i + self.batch_size]))
        return embeddings


if __name__ == "__main__":
    embed_openai = EmbedOpenAI()
    print(embed_openai.batch_embed(["This is a test"]))
