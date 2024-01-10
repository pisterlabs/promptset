import openai

from models.ModelConfig import (
    GPTCard,
    Message,
    Messages,
    SingleFlashcardGeneratorConfig,
)
from text.GPTInterface import GPTInterface


class SingleFlashcardGeneratorMock(GPTInterface):
    def __call__(self, text: str, user_id: str) -> GPTCard:
        return GPTCard(
            question="What is the capital of the United States?",
            answer="Washington D.C.",
        )

    def _generate_messages(self) -> Messages:
        return []


class SingleFlashcardGenerator(GPTInterface):
    _model_config: SingleFlashcardGeneratorConfig

    def __init__(
        self,
        model_config: SingleFlashcardGeneratorConfig,
        openai_api_key: str,
    ) -> None:
        openai.api_key = openai_api_key
        self._model_config = model_config

    def __call__(self, text: str, user_id: str) -> GPTCard:
        messages = self._generate_messages(text)
        completion = self._get_completion(messages, user_id)
        card = self._postprocess(completion)

        return card

    def _postprocess(self, completion: str) -> GPTCard:
        split_completion = completion.split("\n")

        question = split_completion[0].replace("Q: ", "")
        answer = split_completion[1].replace("A: ", "")

        card = GPTCard(question=question, answer=answer)

        return card

    def _generate_messages(self, text: str) -> Messages:
        system_message = self._model_config.system_message
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=text),
        ]

        return messages
