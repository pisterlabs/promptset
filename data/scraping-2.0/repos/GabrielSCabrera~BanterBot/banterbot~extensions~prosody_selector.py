import logging
import re
from typing import Optional

from banterbot.config import RETRY_LIMIT
from banterbot.data.enums import ChatCompletionRoles, Prosody
from banterbot.data.prompts import ProsodySelection
from banterbot.models.azure_neural_voice_profile import AzureNeuralVoiceProfile
from banterbot.models.message import Message
from banterbot.models.openai_model import OpenAIModel
from banterbot.models.phrase import Phrase


class ProsodySelector:
    """
    The ProsodySelector class is responsible for managing prosody selection/extraction for specified instances of the
    AzureNeuralVoice class. It uses the OpenAI ChatCompletion API to generate prosody settings for a list of sentences.

    Attributes:
        _model (OpenAIModel): The OpenAI model to be used for generating responses.
        _openai_manager (OpenAIService): An instance of the OpenAIService class.
        _voice (AzureNeuralVoice): An instance of the AzureNeuralVoice class.
        _valid (bool): A flag indicating whether the voice styles are not None.
        _token_counts (dict): A dictionary to cache the maximum number of tokens for a given number of rows.
        _output_patterns (dict): A dictionary to cache the regex patterns matching the expected ChatCompletion output.
        _system (list[Message]): A list of system and user messages to be used as a prompt for the ChatCompletion API.
        _line_pattern (str): A regex pattern that matches one line of expected output for the current model.
    """

    def __init__(self, manager: OpenAIModel, voice: AzureNeuralVoiceProfile) -> None:
        """
        Initializes the ProsodySelector class with a specified OpenAI model and AzureNeuralVoice instance.

        Args:
            manager (OpenAIService): An instance of class OpenAIService to be used for generating responses.
            voice (AzureNeuralVoice): An instance of the AzureNeuralVoice class.
        """
        logging.debug(f"ProsodySelector initialized")
        self._manager = manager
        self._voice = voice
        self._valid = self._voice.style_list is not None
        self._token_counts = {}
        self._output_patterns = {}
        self._init_system()

    def _init_system(self) -> None:
        """
        Prepare the system prompt on instantiation, which is customized on a model-to-model basis, since different
        `OpenAIModel` instances vary in terms of available styles. Also prepares a regex pattern that matches one line
        of expected output for the current model.
        """
        # Convert the different prosody options into
        styles = "\n".join([f"{n+1:02d} {i}" for n, i in enumerate(self._voice.style_list)])
        styledegrees = "\n".join([f"{n+1} {i}" for n, i in enumerate(Prosody.STYLEDEGREES)])
        pitches = "\n".join([f"{n+1} {i}" for n, i in enumerate(Prosody.PITCHES)])
        rates = "\n".join([f"{n+1} {i}" for n, i in enumerate(Prosody.RATES)])
        emphases = "\n".join([f"{n+1} {i}" for n, i in enumerate(Prosody.EMPHASES)])

        self._system = [
            Message(role=ChatCompletionRoles.SYSTEM, content=ProsodySelection.PREFIX.value),
            Message(role=ChatCompletionRoles.USER, content=ProsodySelection.STYLE_USER.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=ProsodySelection.STYLE_ASSISTANT.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=styles),
            Message(role=ChatCompletionRoles.USER, content=ProsodySelection.STYLEDEGREE_USER.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=ProsodySelection.STYLEDEGREE_ASSISTANT.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=styledegrees),
            Message(role=ChatCompletionRoles.USER, content=ProsodySelection.PITCH_USER.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=ProsodySelection.PITCH_ASSISTANT.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=pitches),
            Message(role=ChatCompletionRoles.USER, content=ProsodySelection.RATE_USER.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=ProsodySelection.RATE_ASSISTANT.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=rates),
            Message(role=ChatCompletionRoles.USER, content=ProsodySelection.EMPHASIS_USER.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=ProsodySelection.EMPHASIS_ASSISTANT.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=emphases),
            Message(
                role=ChatCompletionRoles.USER,
                content=ProsodySelection.SUFFIX.value.format(
                    style=len(self._voice.style_list) - 1,
                    styledegree=len(Prosody.STYLEDEGREES) - 1,
                    pitch=len(Prosody.PITCHES) - 1,
                    rate=len(Prosody.RATES) - 1,
                    emphasis=len(Prosody.EMPHASES) - 1,
                ),
            ),
            Message(role=ChatCompletionRoles.USER, content=ProsodySelection.EXAMPLE_USER.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=ProsodySelection.EXAMPLE_ASSISTANT_1.value),
            Message(role=ChatCompletionRoles.ASSISTANT, content=ProsodySelection.EXAMPLE_ASSISTANT_2.value),
        ]

    def select(self, sentences: list[str], context: Optional[str] = None, system: Optional[str] = None) -> str:
        """
        Extracts prosody settings for a list of sentences by asking the OpenAI ChatCompletion API to pick a set of
        options. The prompt is set up to force the model to return an exact number of tokens with dummy text preceding
        it in order to yield consistent results efficiently.

        Args:
            sentences (list[str]): The list of sentences to be processed.
            context (Optional[str]): Useful prior conversational context originating from the same response.
            system (Optional[str]): A system prompt to assist the ChatCompletion in picking reactions.

        Returns:
            str: The randomly selected option.
        """
        for i in range(RETRY_LIMIT):
            # Attempt several different sentence splits in order to modify the input on retry -- significantly reduces
            # the chance of raising a `FormatMismatchError` Exception. `RETRY_LIMIT` is defined in the config file.
            if i == 0:
                phrases = self._split_sentences(sentences=sentences)
            elif i == 1:
                phrases = " ".join(sentences).split(".")
            else:
                phrases = [" ".join(sentences)]

            messages = self._get_messages(phrases=phrases, system=system, context=context)

            response = self._manager.prompt(
                messages=messages,
                split=False,
                temperature=0.0,
                top_p=1.0,
                max_tokens=self._get_max_tokens(len(phrases)),
            )
            processed, outputs = self._process_response(phrases, response)

            if processed is not None:
                break

        if processed is None:
            processed = [Phrase(text=sentence, voice=self._voice) for sentence in sentences]
        else:
            outputs = [
                ProsodySelection.PROMPT.value.format(len(phrases), "\n".join(phrases)),
                ProsodySelection.DUMMY.value.format(len(phrases)),
                "\n".join(outputs),
            ]

        return processed, outputs

    def _get_max_tokens(self, N: int) -> int:
        """
        Returns the maximum number of tokens for the specified number of rows of six-digit numbers. Caches all
        calculated values.

        Args:
            N (int): The number of rows.

        Returns:
            int: The maximum number of tokens.
        """
        # Count the number of tokens for the specified `N`.
        if N not in self._token_counts:
            dummy = "012345\n" * (N - 1) + "012345"
            self._token_counts[N] = self._manager.count_tokens(dummy)

        return self._token_counts[N]

    def _get_output_pattern(self, N: int) -> int:
        """
        Returns a compiled regex pattern matching the expected ChatCompletion output for a given `N`, or number of
        phrases. Caches all calculated values.

        Args:
            N (int): The number of phrases.

        Returns:
            int: A compiled regex pattern.
        """
        # Compile a regex pattern that matches `N` lines of expected output from ChatCompletion's prosody evaluation.
        if N not in self._output_patterns:
            self._output_patterns[N] = re.compile(r"\d{6}\n" * (N - 1) + r"\d{6}")

        return self._output_patterns[N]

    def _get_messages(
        self, phrases: list[str], system: Optional[str] = None, context: Optional[str] = None
    ) -> list[Message]:
        """
        Inserts the system prompt, user prompt, prefix, suffix, and a dummy message mimicking a successful interaction
        with the ChatCompletion API, into the list of messages.

        Args:
            phrases (list[str]): The list of phrases to be processed.
            context (Optional[str]): Useful prior conversational context originating from the same response.
            system (Optional[str]): A system prompt to assist the ChatCompletion in picking reactions.

        Returns:
            list[Message]: The enhanced list of messages.
        """
        messages = self._system.copy()

        if system:
            messages.append(
                Message(role=ChatCompletionRoles.SYSTEM, content=ProsodySelection.CHARACTER.value.format(system))
            )

        if context:
            messages.append(
                Message(role=ChatCompletionRoles.SYSTEM, content=ProsodySelection.CONTEXT.value.format(context))
            )

        messages.append(
            Message(
                role=ChatCompletionRoles.USER,
                content=ProsodySelection.PROMPT.value.format(len(phrases), "\n".join(phrases)),
            )
        )
        messages.append(
            Message(role=ChatCompletionRoles.ASSISTANT, content=ProsodySelection.DUMMY.value.format(len(phrases)))
        )

        return messages

    def _process_response(self, phrases: list[str], response: str) -> list[Phrase]:
        """
        Given a response from the ChatCompletion API using the `_prompt` instance attribute, parses one sub-sentence and
        returns an instance of class `Phrase`.

        Args:
            phrases (str): The string divided into sub-sentences (phrases).
            response (str): The ChatCompletion response to be processed.

        Returns:
            list[Phrase]: A processed list of instances of class `Phrase`.
        """
        processed = []
        pattern = self._get_output_pattern(len(phrases))
        if re.fullmatch(pattern, response) is not None:
            outputs = re.findall(self._get_output_pattern(1), response)
            for output, phrase in zip(outputs, phrases):
                processed.append(self._create_phrase(output, phrase))
            return processed, outputs
        else:
            return None, None

    def _create_phrase(self, output: str, phrase: str) -> Phrase:
        """
        Given an output from the ChatCompletion API and a phrase, creates an instance of class `Phrase`.

        Args:
            output (str): The output from the ChatCompletion API.
            phrase (str): The phrase to be processed.

        Returns:
            Phrase: An instance of class `Phrase`.
        """
        # style = self._voice.style_list[min(int(output[:2]), len(self._voice.style_list)) - 1]

        indices = {
            "style": [self._voice.style_list, str()],
            "styledegree": [list(Prosody.STYLEDEGREES.values()), str()],
            "pitch": [list(Prosody.PITCHES.values()), str()],
            "rate": [list(Prosody.RATES.values()), str()],
            "emphasis": [list(Prosody.EMPHASES.values()), str()],
        }

        kwargs = {}

        for n, (key, value) in enumerate(indices.items()):
            try:
                if n == 0:
                    idx = int(output[:2])
                else:
                    idx = int(output[n + 1])

                if 0 <= idx < len(value[0]):
                    value[1] = value[0][idx]
                else:
                    logging.debug(f"ProsodySelector failed to parse {key} from {output[n + 1]} as valid index")
            except ValueError:
                logging.debug(f"ProsodySelector failed to parse {key} from {output[n + 1]} as integer")

            kwargs[key] = value[1]

        return Phrase(
            text=phrase,
            voice=self._voice,
            **kwargs,
        )

    def _split_sentences(self, sentences: list[str]) -> tuple[list[str], int]:
        """
        Given a list of sentences, splits them on certain types of punctuation (defined in `config.py`) into smaller
        phrases.

        Args:
            sentences (list[Message]): The list of sentences to be processed.

        Returns:
            list[str]: A list of sub-sentences divided on the specified punctuation delimiters.
            int: The number of tokens expected in the ChatCompletion response.
        """
        phrases = []

        for sentence in sentences:
            result = re.split(Prosody.PHRASE_PATTERN, sentence)
            processed = []
            for phrase in result:
                if phrase := phrase.strip():
                    if (not re.match(Prosody.PHRASE_PATTERN, phrase) and phrase.count(" ") > 1) or not processed:
                        processed.append(phrase)
                    else:
                        processed[-1] += phrase
            phrases += processed

        return phrases
