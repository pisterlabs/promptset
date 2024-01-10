"""
A module for general logic for wrapping OpenAI endpoints.
"""
import abc
from collections.abc import Iterable, Mapping
from copy import deepcopy

from ..util.typing_util import SupportedOpenAIClassesType
from ..util.validation_util import validate_openai_class

# These are keys that might exist in the OpenAI request object that we don't
# want to send to Mona. A sort of blacklist of keys.
INPUT_KEYS_TO_POP = ("api_key", "organization")


class OpenAIEndpointWrappingLogic(metaclass=abc.ABCMeta):
    """
    An abstract class used for wrapping OpenAI endpoints. Each child of this
    class must implement several key logics for wrapping specific OpenAI
    endpoints' logic.

    Note that subclasses are not the actual wrappers of the OpenAI endpoint
    classes, but provide functions to create the actual wrapper classes.
    """

    def __init__(self, specs):
        self._specs = specs
        self._analysis_functions = {
            "privacy": self._get_full_privacy_analysis,
            "textual": self._get_full_textual_analysis,
            "profanity": self._get_full_profainty_analysis,
        }

    def wrap_class(self, openai_class) -> SupportedOpenAIClassesType:
        """
        Returns a monitored class wrapping the given openai class, enriching it
        with specific capabilities to be used by an inhereting monitored class.
        """
        validate_openai_class(openai_class, self._get_endpoint_name())

        class WrapperClass(openai_class):
            # TODO(itai): Have a smarter way to "import" all the methods to
            #   this class instead of just copying them.
            @classmethod
            def _get_full_analysis(
                cls, input: Mapping, response: Mapping
            ) -> dict:
                return self.get_full_analysis(input, response)

            @classmethod
            def _get_clean_message(cls, message: Mapping) -> Mapping:
                return self.get_clean_message(message)

            @classmethod
            def _get_stream_delta_text_from_choice(
                cls, choice: Mapping
            ) -> str:
                return self.get_stream_delta_text_from_choice(choice)

            @classmethod
            def _get_final_choice(cls, text: str) -> dict:
                return self.get_final_choice(text)

            @classmethod
            def _get_all_prompt_texts(cls, request: Mapping) -> Iterable[str]:
                return self.get_all_prompt_texts(request)

            @classmethod
            def _get_all_response_texts(
                cls, response: Mapping
            ) -> Iterable[str]:
                return self.get_all_response_texts(response)

        return type(
            f"Monitored{self._get_endpoint_name()}", (WrapperClass,), {}
        )

    @abc.abstractmethod
    def _get_endpoint_name(self) -> str:
        """
        Returns the name of the OpenAI endpoint that is being wrapped.
        """
        pass

    @abc.abstractmethod
    def _internal_get_clean_message(self, message: Mapping) -> Mapping:
        """
        This method will be called in child classes for specific message
        cleaning logic.
        """
        pass

    def get_clean_message(self, message: Mapping) -> Mapping:
        """
        Given a mona message, returns a "clean" message in the sense that it
        will not hold any information that shouldn't be exported to Mona
        (e.g., actual prompts).
        """
        new_message = deepcopy(message)
        for key in INPUT_KEYS_TO_POP:
            new_message["input"].pop(key, None)
        return self._internal_get_clean_message(new_message)

    def get_full_analysis(self, input: Mapping, response: Mapping) -> dict:
        """
        Returns a dict mapping each analysis type to all related analysis
        fields for the given prompt and answers according to the given
        specs (if no "analysis" spec is given - return result for all
        analysis types).

        TODO(itai): Consider propogating the specs to allow the user to
            choose specific anlyses to be made from within each analysis
            category.
        """
        return {
            x: self._analysis_functions[x](input, response)
            for x in self._analysis_functions
            if self._specs.get("analysis", {}).get(x, True)
        }

    @abc.abstractmethod
    def _get_full_privacy_analysis(
        self, input: Mapping, response: Mapping
    ) -> dict:
        """
        Returns a dictionary with all calculated privacy analysis params.
        """
        pass

    @abc.abstractmethod
    def _get_full_textual_analysis(
        self, input: Mapping, response: Mapping
    ) -> dict:
        """
        Returns a dictionary with all calculated textual analysis params.
        """
        pass

    @abc.abstractmethod
    def _get_full_profainty_analysis(
        self, input: Mapping, response: Mapping
    ) -> dict:
        """
        Returns a dictionary with all calculated profanity analysis params.
        """
        pass

    @abc.abstractmethod
    def get_stream_delta_text_from_choice(self, choice: Mapping) -> str:
        """
        Given a stream response "choice", returns the text from that choice.
        """
        pass

    @abc.abstractmethod
    def get_final_choice(self, text: str) -> dict:
        """
        Returns a dictionary for a "choice" object as it would have been
        received from OpenAI's API that holds the given text as the content.
        """
        pass

    @abc.abstractmethod
    def get_all_prompt_texts(self, request: Mapping) -> Iterable[str]:
        """
        Given a request object, returns all the prompt texts from that
        request.
        """
        pass

    @abc.abstractmethod
    def get_all_response_texts(self, response: Mapping) -> Iterable[str]:
        """
        Given a response object, returns all the possible response texts.
        """
        pass
