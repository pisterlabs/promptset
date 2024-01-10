import json
from typing import Dict, Optional, List, Callable, Tuple, Any, Union

from typeguard import typechecked

from chatmanager.util import num_tokens_from_messages
from chatmanager.config import ChatSetup


class ChatMessage:
    """Construct the message for single interaction

    Attributes:
        repo: store the messages, order sensitive
    """

    def __init__(self) -> None:
        self.repo: List[Dict[str, str]] = list()

    def set_repo(self, index: int, message: Dict[str, str]) -> None:
        self.repo[index] = message

    def clear(self) -> None:
        self.repo = list()

    def push_system(self, msg: str) -> None:
        self.push_msg({"role": "system", "content": msg})

    def push_user(self, msg: str) -> None:
        self.push_msg({"role": "user", "content": msg})

    def push_assistant(self, msg: str) -> None:
        self.push_msg({"role": "assistant", "content": msg})

    @typechecked
    def push_msg(self, message: Union[List[Dict[str, str]], Dict[str,
                                                                 str]]) -> None:
        if isinstance(message, dict):
            self.repo.append(message)
        elif isinstance(message, list):
            self.repo.extend(message)
        else:
            raise TypeError("message must be a dict or a list of dict")

    def del_msg(self, delete_checker: Callable[[Dict[str, str]], bool]) -> None:
        """ Delete the entries that satisfy delete_checker (i.e. return True) """

        self.repo = list(filter(lambda x: not delete_checker(x), self.repo))

    def drain(self) -> List[Dict[str, str]]:
        return self.repo

    def token_usage(self) -> int:
        return num_tokens_from_messages(self.drain(), ChatSetup.model)

    def __str__(self) -> str:
        return str(self.repo)


class ChatResponse:
    """Parse the response from openai

    {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
            "role": "assistant"
          }
        }
      ],
      "created": 1677664795,
      "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
      "model": "gpt-3.5-turbo-0613",
      "object": "chat.completion",
      "usage": {
        "completion_tokens": 17,
        "prompt_tokens": 57,
        "total_tokens": 74
      }
    }

    Attributes:
        response: The response from openai

    Methods:
        token_usage: Get the number of tokens used
        choice_num: Get the number of choices
        get_msg: Get the message of a choice
    """

    def __init__(self, response) -> None:
        self.response = response

        self.choices: List[Dict[str, str]] = response["choices"]
        self.created: int = response["created"]
        self.id: str = response["id"]
        self.model: str = response["model"]
        self.usage: Dict[str, int] = response["usage"]

    def token_usage(self, choice: str = 'total_tokens') -> int:
        assert (choice
                in ['completion_tokens', 'prompt_tokens', 'total_tokens'])
        return self.response["usage"][choice]

    def choice_num(self) -> int:
        return len(self.response["choices"])

    def get_choice(self, num: int) -> Optional[Dict[str, str]]:
        if num >= self.choice_num():
            # TODO: throw error
            return None

        return self.choices[num]

    def get_msg(self, choice: int = 0) -> Optional[str]:
        if choice >= self.choice_num():
            return None

        return self.response["choices"][choice]["message"]["content"]


class Session:
    """Log the interaction between user and openai

    Store each message and response in a list

    Attributes:
        name: The name of the session
        repo: The list of (message, response) tuples
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.repo: List[Tuple[ChatMessage, Optional[ChatResponse]]] = list()

    def push(self, msg: ChatMessage, response: Optional[ChatResponse]) -> None:
        self.repo.append((msg, response))

    def export(
        self, export_processor: Callable[[ChatMessage, Optional[ChatResponse]],
                                         Any]
    ) -> str:
        """ Export the session to a string

        Args:
            export_processor: A function that process each (message, response) pair
        """

        lst = list(map(lambda x: export_processor(x[0], x[1]), self.repo))
        return json.dumps(lst, indent=4)
