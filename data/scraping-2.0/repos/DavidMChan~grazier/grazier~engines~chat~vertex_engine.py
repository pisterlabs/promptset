try:
    from vertexai.preview.language_models import ChatModel, InputOutputTextPair
except ImportError:
    ChatModel = None
    InputOutputTextPair = None

from typing import Any, Dict, List

from ratelimit import limits, sleep_and_retry

from grazier.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from grazier.utils.python import singleton


class VertexLLMEngine(LLMChat):
    def __init__(self, model: str, **kwargs: Dict[str, Any]) -> None:
        super().__init__(device="api")

        if ChatModel is None:
            raise ImportError("Please install the Vertex AI SDK to use this LM engine.")

        self._model = ChatModel.from_pretrained(model)
        self._parameters = {
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        } | kwargs

    @sleep_and_retry  # type: ignore
    @limits(  # type: ignore
        calls=40, period=60
    )  # This is the default rate limit for Vertex AI (actual rate limit is 60 calls per minute, but we'll be conservative)
    def _rate_limited_model_predict(self, info, **kwargs: Any) -> Any:
        context, examples, prompt = info
        chat = self._model.start_chat(context=context, examples=examples, **kwargs)
        response = chat.send_message(prompt).text

        return response

    def call(self, conversation: Conversation, n_completions: int = 1, **kwargs: Any) -> List[ConversationTurn]:
        # Start the chat
        system_turns = [c for c in conversation.turns if c.speaker == Speaker.SYSTEM]
        context = system_turns[-1].text if system_turns else ""
        non_system_turns = [c for c in conversation.turns if c.speaker != Speaker.SYSTEM]

        # Assert that the non-system turns alternate between the user and the agent
        for idx, turn in enumerate(non_system_turns):
            if idx % 2 == 0:
                assert turn.speaker == Speaker.USER
            else:
                assert turn.speaker == Speaker.AI
        if len(non_system_turns) > 1:
            assert non_system_turns[-1].speaker == Speaker.USER
            assert InputOutputTextPair is not None
            # Build the examples
            examples = [
                InputOutputTextPair(input_text=non_system_turns[i].text, output_text=non_system_turns[i + 1].text)
                for i in range(0, len(non_system_turns) - 1, 2)
            ]
        else:
            examples = []

        # Normalize kwargs from openai to vertexai (some common parameters are different)
        kwargs = (
            self._parameters
            | {
                "max_output_tokens": kwargs.pop("max_output_tokens", kwargs.pop("max_tokens", 256)),
                "temperature": kwargs.pop("temperature", 1.0),
            }
            | kwargs
        )

        return [
            ConversationTurn(
                self._rate_limited_model_predict((context, examples, non_system_turns[-1].text), **kwargs),
                speaker=Speaker.AI,
            )  # type: ignore
            for _ in range(n_completions)
        ]

    @staticmethod
    def is_configured() -> bool:
        # Check to see if the Vertex AI SDK is installed, and if so, if the user has configured their credentials.
        if ChatModel is None or InputOutputTextPair is None:
            return False

        # Check to see if the user has configured their google cloud credentials.
        try:
            from google.auth import default
        except ImportError:
            return False

        try:
            default()
        except Exception:
            return False

        return True


@register_engine
@singleton
class PaLMEngine(VertexLLMEngine):
    name = ("PaLM", "palm")

    def __init__(self) -> None:
        super().__init__("chat-bison@001")
