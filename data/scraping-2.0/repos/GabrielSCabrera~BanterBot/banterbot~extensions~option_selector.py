import logging

from banterbot.data.enums import ChatCompletionRoles
from banterbot.data.prompts import OptionSelectorPrompts
from banterbot.models.message import Message
from banterbot.models.openai_model import OpenAIModel
from banterbot.services.openai_service import OpenAIService


class OptionSelector:
    """
    The OptionSelector class facilitates evaluating and selecting the most suitable option from a set of provided
    options given a conversational context.

    This class enhances the capabilities of the base `OpenAIService` by providing a mechanism for option assessment for
    potential responses. The options provided can represent any category or attribute, such as emotional tones, topics,
    sentiment, etc., thus allowing for a variety of uses.

    The class accepts three main parameters: a list of options (strings), a prompt, and an initial system message. The
    system message sets the context for the OptionSelector's task, while the prompt provides a guideline for the
    evaluation. The most suitable option is then selected based on the specified conversational context.

    Example: Emotional Tone Selection

        options = ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly"]
        prompt = "Choose the most suitable tone/emotion for the assistant's upcoming response."
        system = (
            "You are an Emotional Tone Evaluator. Given conversational context, you analyze and select the most "
            "appropriate tone/emotion that the assistant is likely to use next."
        )

    This example showcases the OptionSelector as an "Emotional Tone Evaluator". The options are different emotional
    tones. Based on a conversational context, OptionSelector selects the most suitable tone for the assistant's next
    response.
    """

    def __init__(self, model: OpenAIModel, options: list[str], system: str, prompt: str):
        """
        Initialize the OptionSelector with the specified model, options, system message, prompt, and optional seed.

        Args:
            model (OpenAIModel): The OpenAI model to be used for generating responses.
            options (list[str]): A list of strings representing the options to be evaluated.
            system (str): The initial system message that sets the context for the OptionSelector's task.
            prompt (str): The prompt that provides a guideline for the evaluation.
        """
        logging.debug(f"OptionSelector initialized")
        self._options = options
        self._system = system
        self._prompt = prompt

        self._openai_manager = OpenAIService(model=model)
        self._system_processed = self._init_system_prompt()

    def select(self, messages: list[Message]) -> str:
        """
        Select an option by asking the OpenAI ChatCompletion API to pick an answer. The prompt is set up to force the
        model to return a single token with dummy text preceding it in order to yield consistent results in an efficient
        way.

        Args:
            messages (list[Message]): The list of messages to be processed.

        Returns:
            str: The randomly selected option.
        """
        messages = self._insert_messages(messages)
        response = self._openai_manager.prompt(messages=messages, split=False, temperature=0.0, top_p=1.0, max_tokens=1)
        try:
            selection = self._options[int(response) - 1]
        except:
            selection = None

        logging.debug(f"OptionSelector selected option: `{selection}`")
        return selection

    def _init_system_prompt(self) -> str:
        """
        Initialize the system prompt by combining the system message and the options.

        Returns:
            str: The processed system prompt.
        """
        options = ", ".join(f"{n+1} {option}" for n, option in enumerate(self._options))
        system_prompt = f"{self._system} {OptionSelectorPrompts.PREFIX.value}{options}"
        return system_prompt

    def _insert_messages(self, messages: list[Message]) -> list[Message]:
        """
        Insert the system prompt, user prompt, prefix, suffix, and a dummy message mimicking a successful interaction
        with the ChatCompletion API, into the list of messages.

        Args:
            messages (list[Message]): The list of messages to be processed.

        Returns:
            list[Message]: The enhanced list of messages.
        """
        prefix = Message(role=ChatCompletionRoles.SYSTEM, content=self._system_processed)
        suffix = Message(role=ChatCompletionRoles.USER, content=self._prompt)
        dummy_message = Message(role=ChatCompletionRoles.ASSISTANT, content=OptionSelectorPrompts.DUMMY.value)
        messages = [prefix] + messages + [suffix, dummy_message]
        return messages
