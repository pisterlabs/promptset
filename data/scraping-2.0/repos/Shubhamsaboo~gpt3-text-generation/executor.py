from urllib import response
from jina import Executor, requests
from docarray import DocumentArray, Document
import openai
from typing import Optional


class Gpt3TextGeneration(Executor):
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = 20,
        temperature: Optional[float] = 0.7,
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.api_key = api_key 
        self.max_tokens = max_tokens
        self.temperature = temperature

    @requests(on="/complete")
    def completion(self, docs: DocumentArray, **kwargs):

        for i, d in enumerate(docs):
            docs[i].text = self._complete(d)

    def _complete(self, doc: Document):
        openai.api_key = self.api_key if self.api_key else doc.tags['api_key']

        completion = openai.Completion.create(
            model="text-davinci-002",
            prompt=doc.text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        response = completion['choices'][0]['text'].strip()

        return response
