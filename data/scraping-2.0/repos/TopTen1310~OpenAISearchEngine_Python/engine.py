import os
import openai
import numpy as np

class Engine:
    def __init__(self) -> None:
        self.openai = openai
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "text-davinci-003"

    def set_model(self, model):
        self.model = model

    def get_model_list(self):
        response = self.openai.Model.list()
        return list(map(lambda x: x["id"], response["data"]))

    def retrieve_model(self, model):
        return self.openai.Model.retrieve(model)

    def get_completion(self, prompt):
        return self.openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=7,
            temperature=0
        )

    def get_embedding(self, texts):
        response = self.openai.Embedding.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return list(map(lambda x: x["embedding"], response["data"]))
