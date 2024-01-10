import os

from openai import OpenAI

from llm_model.llm_model import LlmModel
from model.llm_model_response import LlmModelResponse


class LlmModelOpenAI(LlmModel):
    def __init__(self, model_name):
        super().__init__(model_name)

        os.environ["OPENAI_API_KEY"]  # Throws exception if key is not in OPENAI_API_KEY

        self.openai_client = OpenAI()

    def get_model_response(self, system: str, user_query: str) -> LlmModelResponse:
        prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query}
        ]

        llm_full_response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=prompt
        )

        return LlmModelResponse(
            llm_full_response.choices[0].message.content,
            llm_full_response.__str__())
