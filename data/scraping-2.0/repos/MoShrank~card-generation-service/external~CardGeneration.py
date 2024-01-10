from abc import ABC, abstractmethod
from typing import Any, List

import openai

from external.gpt import get_chatgpt_completion
from models.ModelConfig import CardGenerationConfig, Example, Message
from models.Note import GPTCard
from util.error import retry_on_exception


class CardGenerationInterface(ABC):
    @abstractmethod
    def __call__(self, text: str, user_id: str) -> List[GPTCard]:
        pass


class CardGenerationMock(CardGenerationInterface):
    def __call__(self, text: str, user_id: str) -> List[GPTCard]:
        return [
            GPTCard(
                question="What is the capital of the United States?",
                answer="Washington D.C.",
            ),
            GPTCard(
                question="What is the capital of the United States?",
                answer="Washington D.C.",
            ),
            GPTCard(
                question="What is the capital of the United States?",
                answer="Washington D.C.",
            ),
        ]


class CardGeneration(CardGenerationInterface):
    _model_config: CardGenerationConfig

    def __init__(self, model_config: CardGenerationConfig, openai_api_key: str) -> None:
        openai.api_key = openai_api_key
        self._model_config = model_config

    def __call__(self, text: str, user_id: str) -> List[GPTCard]:
        preprocessed_text = self.preprocess(text)
        prompt = self._generate_prompt(preprocessed_text)
        completion = self._generate_cards(prompt, user_id)
        cards = self.postprocess(completion)

        return cards

    def preprocess(self, text: str) -> str:
        return text.replace("\n\n", "\n")

    def postprocess(self, completion: str) -> Any:
        qas = completion.split("\n\n")

        parsed_qas = []
        for qa in qas:
            split_qa = qa.split("\n")
            if len(split_qa) == 2:
                question = split_qa[0].strip().replace("Q: ", "")
                answer = split_qa[1].strip().replace("A: ", "")
                card = GPTCard(question=question, answer=answer)
                parsed_qas.append(card)

        return parsed_qas

    @retry_on_exception(exception=Exception)
    def _generate_cards(self, prompt: str, user_id: str) -> str:
        messages = [Message(role="user", content=prompt)]
        completion = get_chatgpt_completion(
            self._model_config.parameters, messages, user_id
        )

        return completion

    def _get_qa_text(self, qa: GPTCard) -> str:
        return "Q: " + qa.question + "\nA: " + qa.answer

    def _get_example_text(
        self, example: Example, stop_sequence: str, card_prefix: str, note_prefix: str
    ) -> str:
        examples_text = "\n\n".join([self._get_qa_text(qa) for qa in example.cards])

        example_text = (
            note_prefix
            + "\n"
            + example.note
            + "\n"
            + card_prefix
            + "\n"
            + examples_text
            + "\n\n"
            + stop_sequence
        )

        return example_text

    def _generate_prompt(self, text: str) -> str:
        examples_text = ""

        stop_sequence = (
            self._model_config.parameters.stop_sequence[0]
            if self._model_config.parameters.stop_sequence
            else ""
        )

        if len(self._model_config.examples):
            examples_text = "\n\n".join(
                [
                    self._get_example_text(
                        example,
                        stop_sequence,
                        self._model_config.card_prefix,
                        self._model_config.note_prefix,
                    )
                    for example in self._model_config.examples
                ]
            )

            examples_text += "\n\n" + stop_sequence

        prompt = (
            self._model_config.prompt_prefix
            + "\n\n"
            + examples_text
            + self._model_config.note_prefix
            + "\n"
            + text
            + "\n\n"
            + self._model_config.card_prefix
            + "\n"
        )

        return prompt
