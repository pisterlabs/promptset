import json
import logging
from dataclasses import dataclass
from functools import cached_property

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from rule_articulation.secrets import get_openai_key
from rule_articulation.task_model import (
    TaskDescription,
    LabelledInput,
    format_labelled_input,
)

logger = logging.getLogger(__name__)
client = OpenAI(**get_openai_key())
total_tokens_used = 0


@dataclass
class EvaluationReport:
    task: TaskDescription
    test_data: list[LabelledInput]
    responses: list[bool | None]

    @cached_property
    def fraction_correct(self) -> float:
        return sum(
            [
                1
                for labelled_input, response in zip(self.test_data, self.responses)
                if response == labelled_input.label
            ]
        ) / len(self.test_data)

    @cached_property
    def mislabelled_responses(self) -> list[tuple[LabelledInput, bool | None]]:
        return [
            (labelled_input, response)
            for labelled_input, response in zip(self.test_data, self.responses)
            if response != labelled_input.label
        ]

    def print(self):
        print(f"Fraction correct: {self.fraction_correct}")
        print(f"Total tokens used: {total_tokens_used}")
        print("Mislabelled responses:")
        for labelled_input, response in self.mislabelled_responses:
            print(f"Input: {labelled_input.input}")
            print(f"Label: {labelled_input.label}")
            print(f"Response: {response}")
            print()


def messages_for_single_prompt(
    system_prompt: str,
    user_prompt: str,
) -> list[ChatCompletionMessageParam]:
    """Just a system prompt and a user prompt"""
    return [
        ChatCompletionSystemMessageParam(
            content=system_prompt,
            role="system",
        ),
        ChatCompletionUserMessageParam(
            content=user_prompt,
            role="user",
        ),
    ]


class TaskEvaluator:
    openai_model: str
    task: TaskDescription
    num_articulations: int

    def __init__(
        self,
        task: TaskDescription,
        openai_model: str,
        num_articulations: int = 10,
    ):
        self.task = task
        self.openai_model = openai_model
        self.num_articulations = num_articulations

    def evaluate(self, test_data: list[LabelledInput]) -> EvaluationReport:
        input_strings = [labelled_input.input for labelled_input in test_data]

        output = [
            self.get_true_or_false_response(self.evaluation_prompt_messages(input))
            for input in input_strings
        ]

        return EvaluationReport(self.task, test_data, output)

    def ask_articulation(self) -> list[str]:
        response = self.send_prompt(
            json=False,
            messages=self.articulation_prompt_messages(),
            n=self.num_articulations,
        )

        articulation = [choice.message.content for choice in response.choices]

        return articulation

    def evaluation_prompt_messages(
        self, input: str
    ) -> list[ChatCompletionMessageParam]:
        return messages_for_single_prompt(
            system_prompt=self.task.get_system_prompt(),
            user_prompt=format_labelled_input(input, "???"),
        )

    def articulation_prompt_messages(self) -> list[ChatCompletionMessageParam]:
        return messages_for_single_prompt(
            system_prompt="""\
You are an excellent and precise classifier that is well-versed in logic and reasoning.
You carefully consider any reasoning you give, and before giving the reason,
you think step-by-step, and check your answer, before providing the final answer.
You collocutor is equally precise and logical, and is well-informed.
""",
            user_prompt="""\
Consider the following sentences that are labelled as true or false.
Deduce the rule that determines whether a sentence is true or false.
Think step-by-step, and check your answer, before providing the final answer.

"""
            + "\n\n".join(
                [
                    format_labelled_input(labelled_input.input, labelled_input.label)
                    for labelled_input in self.task.example_labelled_inputs
                ]
            ),
        )

    def send_prompt(
        self,
        messages: list[ChatCompletionMessageParam],
        json: bool = True,
        n: int = 1,
    ) -> ChatCompletion:
        global total_tokens_used
        if len(messages) == 0:
            raise ValueError("Must provide at least one message")
        elif len(messages) == 2:
            logger.debug("System prompt: %s", messages[0]["content"])
            logger.debug("User prompt: %s", messages[1]["content"])
        else:
            logger.debug("Prompt: %s", messages[-1]["content"])

        additional_kwargs = {"response_format": {"type": "json_object"}} if json else {}

        response = client.chat.completions.create(
            # model="gpt-4-1106-preview",
            model=self.openai_model,
            messages=messages,
            n=n,
            **additional_kwargs,
        )
        total_tokens_used += response.usage.total_tokens
        logger.debug("Response: %s", response.choices[0].message.content)

        return response

    def get_true_or_false_response(
        self, messages: list[ChatCompletionMessageParam], json_key: str = "label"
    ) -> bool | None:
        """Basically, send_prompt + parsing"""
        response = self.send_prompt(messages)
        content = json.loads(response.choices[0].message.content)

        match content.get(json_key):
            case True:
                return True
            case False:
                return False
            case _:
                print(f"Invalid response: {content}")
                return None
