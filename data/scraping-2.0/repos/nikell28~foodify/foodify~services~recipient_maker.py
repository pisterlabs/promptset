import openai

from foodify.config import config
from foodify.models.recipient import RecipientInput, Recipient


class _ChatGPTRecipientMaker:
    def get_chatgpt_answer(self, message: list[dict]) -> str:
        api_key = config.openai_api_key
        openai.api_key = api_key
        model = "gpt-3.5-turbo"

        response = openai.ChatCompletion.create(
            model=model,
            messages=message,
        )

        recipient = response.choices[0]["message"]["content"].split("\n")
        return recipient

    def _get_recipe(self, recipe_description: str) -> str:
        user_input = [
            {"role": "system", "content": "Ты известный шеф повар."},
            {"role": "user", "content": ""},
        ]
        user_input[1]["content"] = recipe_description
        response = self.get_chatgpt_answer(user_input)
        response = "\n".join(response)
        return response


class PromtCreator:
    def get_promt(self, recipe_input: RecipientInput) -> str:
        promt = config.promt

        return promt  # type: ignore


class RecipientMaker(_ChatGPTRecipientMaker):
    async def get_recipe(self, recipe_input: RecipientInput) -> Recipient:
        promt = PromtCreator().get_promt(recipe_input)
        recipient = self._get_recipe(promt)
        return Recipient(
            description=recipient,
        )
