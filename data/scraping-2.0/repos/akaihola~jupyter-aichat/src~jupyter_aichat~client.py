from itertools import takewhile
from typing import Union, Iterable, NamedTuple

import openai

from jupyter_aichat.output import output
from jupyter_aichat.authentication import authenticate
from jupyter_aichat.api_types import Message, PromptUsage, Request, Response
from jupyter_aichat.schedule import Schedule
from jupyter_aichat.tokens import num_tokens_from_messages


class ScheduledMessage(NamedTuple):
    message: Message
    schedule: Schedule


class Conversation:
    MAX_TOKENS = 4096
    MODEL = "gpt-3.5-turbo"

    def __init__(self) -> None:
        self.transmissions: list[Union[Request, Response]] = []
        self.system_message_schedules: list[ScheduledMessage] = []

    def say_and_listen(self, text: str) -> None:
        authenticate()
        request_message: Message = {"role": "user", "content": text}
        prompt_tokens = num_tokens_from_messages([request_message])
        prompt: Request = {
            "choices": [{"message": request_message}],
            "usage": {
                "total_tokens": self.total_tokens + prompt_tokens,
            },
        }
        self.add_scheduled_system_messages()
        self.transmissions.append(prompt)
        # https://platform.openai.com/docs/api-reference/chat/create
        response = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model=self.MODEL,
            messages=self.get_messages(max_tokens=self.MAX_TOKENS),
        )
        prompt["usage"]["total_tokens"] = response.usage.prompt_tokens
        self.transmissions.append(response.to_dict_recursive())
        response_message = response["choices"][0]["message"]
        output(response_message["content"].strip())

    def get_transmissions(self, max_tokens: int) -> list[Union[Request, Response]]:
        """Return the transmissions that fit within the token budget.

        :param max_tokens: The maximum number of tokens to use.
        :return: The transmissions that fit within the token budget.

        """
        if self.total_tokens <= max_tokens:
            return self.transmissions
        num_transmissions = len(self.transmissions)
        num_system_messages, system_message_tokens = self._get_initial_system_messages()
        for index in range(num_system_messages, num_transmissions):
            tail_tokens = self.get_tokens_for_slice(index, num_transmissions)
            if system_message_tokens + tail_tokens <= max_tokens:
                break
        else:
            index = num_transmissions - 1
            last_message_tokens = self.get_tokens_for_slice(index, num_transmissions)
            if last_message_tokens > max_tokens:
                raise RuntimeError(
                    f"The last message has {last_message_tokens} tokens,"
                    f" more than the maximum of {max_tokens}."
                )
            num_system_messages = 0
        return self.transmissions[:num_system_messages] + self.transmissions[index:]

    def _get_initial_system_messages(self) -> tuple[int, int]:
        """Return the number of initial system messages and their total tokens.

        :return: The number of initial system messages and their total tokens.

        """
        system_prompts = list(takewhile(is_system_prompt, self.transmissions))
        if not system_prompts:
            return 0, 0
        last_system_prompt = system_prompts[-1]
        return len(system_prompts), last_system_prompt["usage"]["total_tokens"]

    def get_tokens_for_slice(self, start: int, stop: int) -> int:
        """Return the total number of tokens in the slice.

        :param start: The start index of the slice.
        :param stop: The stop index of the slice.
        :return: The total number of tokens in the slice.

        """
        if not self.transmissions:
            return 0
        if start == 0:
            return self.transmissions[stop - 1]["usage"]["total_tokens"]
        return (
            self.transmissions[stop - 1]["usage"]["total_tokens"]
            - self.transmissions[start - 1]["usage"]["total_tokens"]
        )

    def get_messages(self, max_tokens: int = 2**63 - 1) -> list[Message]:
        """Return the messages that fit within the token budget.

        :param max_tokens: The maximum number of tokens to use.
        :return: The messages that fit within the token budget.

        """
        return [
            transmission["choices"][0]["message"]
            for transmission in self.get_transmissions(max_tokens)
        ]

    @property
    def current_step(self) -> int:
        """Return the number of completions in the conversation so far.

        :return: The zero-based number of the current step of the conversation.

        """
        return sum(1 for t in self.transmissions if prompt_role_is(t, "assistant"))

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens the conversation corresponds to.

        :return: The total number of tokens.

        """
        if not self.transmissions:
            return 0
        return self.transmissions[-1]["usage"]["total_tokens"]

    def register_system_message(
        self, content: str, schedule: Schedule, skip_if_exists: bool = False
    ) -> None:
        """Add a system message to the conversation.

        :param content: The content of the system message.
        :param schedule: The schedule rule for the system message.
        :param skip_if_exists: Whether to skip adding the message if it already exists.

        """
        if skip_if_exists and any(
            is_system_prompt(prompt)
            and prompt["choices"][0]["message"]["content"] == content
            for prompt in self.transmissions
        ):
            return
        message = Message(role="system", content=content)
        self.system_message_schedules.append(ScheduledMessage(message, schedule))

    def get_scheduled_system_messages(self, step: int) -> Iterable[Request]:
        """Return all any system messages that should be sent at the given step.

        :param step: The number of a step of the conversation.
        :return: The system messages that should be sent at the given step.

        """
        for message, schedule in self.system_message_schedules:
            if not schedule.should_send(step):
                continue
            total_tokens = self.total_tokens + num_tokens_from_messages([message])
            usage = PromptUsage(total_tokens=total_tokens)
            request = Request(choices=[{"message": message}], usage=usage)
            yield request

    def add_scheduled_system_messages(self) -> None:
        """Add scheduled system messages to the conversation.

        Also remove past duplicates of system messages which are scheduled to be sent
        at the current step.

        """
        new_system_prompts = list(self.get_scheduled_system_messages(self.current_step))
        new_system_messages = [
            prompt["choices"][0]["message"]["content"] for prompt in new_system_prompts
        ]
        self.transmissions = [
            xmission
            for xmission in self.transmissions
            if not prompt_role_is(xmission, "system")
            or xmission["choices"][0]["message"]["content"] not in new_system_messages
        ]
        self.transmissions.extend(new_system_prompts)


def prompt_role_is(prompt: Union[Request, Response], role: str) -> bool:
    """Return whether the role of the prompt sender matches the given role.

    :param prompt: The prompt.
    :param role: The role.
    :return: Whether the prompt sender has the given role.

    """
    return prompt["choices"][0]["message"]["role"] == role


def is_system_prompt(prompt: Union[Request, Response]) -> bool:
    """Return whether the prompt is a system prompt.

    :param prompt: The prompt.
    :return: Whether the prompt is a system prompt.

    """
    return prompt_role_is(prompt, "system")
