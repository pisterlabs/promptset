import datetime
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Union

from banterbot import config
from banterbot.data.enums import ChatCompletionRoles
from banterbot.exceptions.format_mismatch_error import FormatMismatchError
from banterbot.extensions.prosody_selector import ProsodySelector
from banterbot.managers.azure_neural_voice_manager import AzureNeuralVoiceManager
from banterbot.managers.openai_model_manager import OpenAIModelManager
from banterbot.models.azure_neural_voice_profile import AzureNeuralVoiceProfile
from banterbot.models.message import Message
from banterbot.models.openai_model import OpenAIModel
from banterbot.paths import chat_logs
from banterbot.services.openai_service import OpenAIService
from banterbot.services.speech_recognition_service import SpeechRecognitionService
from banterbot.services.speech_synthesis_service import SpeechSynthesisService
from banterbot.utils.thread_queue import ThreadQueue


class Interface(ABC):
    """
    Interface is an abstract base class for creating frontends for the BanterBot application. It provides a high-level
    interface for managing conversation with the bot, including sending messages, receiving responses, and updating a
    conversation area. The interface supports both text and speech-to-text input for user messages.
    """

    def __init__(
        self,
        model: OpenAIModel,
        voice: AzureNeuralVoiceProfile,
        languages: Optional[Union[str, list[str]]] = None,
        system: Optional[str] = None,
        tone_model: OpenAIModel = None,
        phrase_list: Optional[list[str]] = None,
        assistant_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the Interface with the specified model and voice.

        Args:
            model (OpenAIModel): The OpenAI model to use for generating responses.
            voice (AzureNeuralVoice): The voice to use for text-to-speech synthesis.
            languages (Optional[Union[str, list[str]]]): The languages supported by the speech-to-text recognizer.
            system (Optional[str]): An initialization prompt that can be used to set the scene.
            tone_model (OpenAIModel): The OpenAI ChatCompletion model to use for tone evaluation.
            phrase_list (list[str], optional): Optionally provide the recognizer with context to improve recognition.
            assistant_name (str, optional): Optionally provide a name for the character.
        """
        logging.debug(f"Interface initialized")

        # Select the default Azure Neural Voice model if not provided.
        self._voice = AzureNeuralVoiceManager.load("Aria") if voice is None else voice
        # Select the default OpenAI ChatCompletion model.
        self._model = OpenAIModelManager.load("gpt-3.5-turbo") if model is None else model
        # Select the default OpenAI ChatCompletion model for tone evaluation.
        self._tone_model = OpenAIModelManager.load("gpt-3.5-turbo") if tone_model is None else tone_model

        # Initialize OpenAI ChatCompletion, Azure Speech-to-Text, and Azure Text-to-Speech components
        self._openai_service = OpenAIService(model=model)
        self._openai_service_tone = OpenAIService(model=tone_model)
        self._speech_recognition_service = SpeechRecognitionService(languages=languages, phrase_list=phrase_list)
        self._speech_synthesis_service = SpeechSynthesisService()

        # Initialize message handling and conversation attributes
        self._messages: list[Message] = []
        self._log_lock = threading.Lock()
        self._log_path = chat_logs / f"chat_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.txt"
        self._listening_toggle = False
        self._listening_active_lock = threading.Lock()
        self._listening_inactive_lock = threading.Lock()

        # Initialize thread management components
        self._thread_queue = ThreadQueue()

        # Initialize assistant name attribute
        self._assistant_name = ChatCompletionRoles.ASSISTANT.value.title() if assistant_name is None else assistant_name

        # Initialize the ProsodySelector.
        self._prosody_selector = ProsodySelector(
            manager=self._openai_service_tone,
            voice=self._voice,
        )

        # Initialize the interruption flag, set to zero.
        self._interrupt = 0

        # Initialize the system message, if provided
        self._system = system
        if self._system is not None:
            message = Message(role=ChatCompletionRoles.SYSTEM, content=system)
            self._messages.append(message)

        # Initialize the subclass GUI
        self._init_gui()

    def interrupt(self, shutdown_time: Optional[int] = None) -> None:
        """
        Interrupts all speech-to-text recognition, text-to-speech synthesis, and OpenAI API streams.

        Args:
            soft (bool): If True, allows the recognizer to keep processing data that was recorded prior to interruption.
            shutdown_time (Optional[int]): The time at which the listener was deactivated.
        """
        logging.debug(f"Interface Interrupted")
        self._interrupt = time.perf_counter_ns() if not shutdown_time else shutdown_time
        self._openai_service.interrupt(kill=True)
        self._openai_service_tone.interrupt(kill=True)
        self._speech_recognition_service.interrupt()
        self._speech_synthesis_service.interrupt()

    def listener_activate(self, name: Optional[str] = None) -> None:
        """
        Activate the speech-to-text listener.

        Args:
            name (Optional[str]): The name of the user sending the message. Defaults to None.
        """
        # Interrupt any currently active ChatCompletion, text-to-speech, or speech-to-text streams
        self.interrupt()
        init_time = time.perf_counter_ns()

        with self._listening_active_lock:
            if not self._listening_toggle and self._interrupt <= init_time:
                self._listen_thread = threading.Thread(
                    target=self._listen,
                    kwargs={"init_time": init_time, "name": name},
                    daemon=True,
                )
                self._listen_thread.start()
                self._listening_toggle = True

    def listener_deactivate(self) -> None:
        """
        Deactivate the speech-to-text listener.
        """
        # Interrupt any currently active ChatCompletion, text-to-speech, or speech-to-text streams
        init_time = time.perf_counter_ns()
        with self._listening_inactive_lock:
            if self._listening_toggle and self._interrupt < init_time:
                self._speech_recognition_service.interrupt()
                self._listening_toggle = False

    def prompt(self, message: str, name: Optional[str] = None) -> None:
        """
        Prompt the bot with the specified user message.

        Args:
            message (str): The message content from the user.
            name (Optional[str]): The name of the user sending the message. Defaults to None.
        """
        # Interrupt any currently active ChatCompletion, text-to-speech, or speech-to-text streams
        self.interrupt()

        # Do not send the message if it is empty.
        if message.strip():
            # Record the time at which the message was initialized in order to account for future interruptions.
            init_time = time.perf_counter_ns()
            self.send_message(message, ChatCompletionRoles.USER, name)
            self._thread_queue.add_task(
                threading.Thread(target=self.respond, kwargs={"init_time": init_time}, daemon=True)
            )

    def send_message(
        self,
        content: str,
        role: ChatCompletionRoles = ChatCompletionRoles.USER,
        name: Optional[str] = None,
        hidden: bool = False,
    ) -> None:
        """
        Send a message from the user to the conversation.

        Args:
            message (str): The message content from the user.
            role (ChatCompletionRoles): The role (USER, ASSISTANT, SYSTEM) associated with the content.
            name (Optional[str]): The name of the user sending the message. Defaults to None.
            hidden (bool): If True, does not display the message in the interface.
        """
        message = Message(role=role, name=name, content=content)
        name = message.name.title() if message.name is not None else ChatCompletionRoles.USER.value.title()
        text = f"{name}: {content}\n\n"
        self._messages.append(message)
        if not hidden:
            self.update_conversation_area(word=text)

    def system_prompt(self, message: str, name: Optional[str] = None) -> None:
        """
        Prompt the bot with the specified message, issuing a command which is not displayed in the conversation area.

        Args:
            message (str): The message content from the user.
        """
        # Do not send the message if it is empty.
        if message.strip():
            # Record the time at which the message was initialized in order to account for future interruptions.
            init_time = time.perf_counter_ns()
            self.send_message(message, ChatCompletionRoles.USER, None, True)
            self._thread_queue.add_task(
                threading.Thread(target=self.respond, kwargs={"init_time": init_time}, daemon=True)
            )

    @abstractmethod
    def update_conversation_area(self, word: str) -> None:
        """
        Update the conversation area with the specified word, and add the word to the chat log.
        This method should be implemented by subclasses to handle updating the specific GUI components.

        Args:
            word (str): The word to add to the conversation area.
        """
        self._append_to_chat_log(word)

    @abstractmethod
    def run(self) -> None:
        """
        Run the frontend application.
        This method should be implemented by subclasses to handle the main event loop of the specific GUI framework.
        """
        ...

    @abstractmethod
    def _init_gui(self) -> None:
        """
        Initialize the graphical user interface for the frontend.
        This method should be implemented by subclasses to create the specific GUI components.
        """
        ...

    def _append_to_chat_log(self, word: str) -> None:
        """
        Updates the chat log with the latest output.

        Args:
            word (str): The word to be added to the conversation area.
        """
        with self._log_lock:
            logging.debug(f"Interface appended new data to the chat log")
            with open(self._log_path, "a+", encoding=config.ENCODING) as fs:
                fs.write(word)

    def respond(self, init_time: int) -> None:
        """
        Get a response from the bot and update the conversation area with the response. This method handles generating
        the bot's response using the OpenAIService and updating the conversation area with the response text using
        text-to-speech synthesis.
        """
        content = ""
        context = []

        # Add the name of the assistant to the conversation area.
        self.update_conversation_area(f"{self._assistant_name}:")

        # Initialize the generator for asynchronous yielding of sentence blocks
        for block in self._openai_service.prompt_stream(messages=self._messages, init_time=init_time):
            phrases, context = self._prosody_selector.select(sentences=block, context=content, system=self._system)
            if phrases is None:
                raise FormatMismatchError()

            for item in self._speech_synthesis_service.synthesize(phrases=phrases, init_time=init_time):
                self.update_conversation_area(item.text)
                content += item.text

        if self._interrupt < init_time and content.strip():
            message = Message(role=ChatCompletionRoles.ASSISTANT, content=content.strip())
            self._messages.append(message)

        self.update_conversation_area("\n\n")

    def _listen(self, init_time: int, name: Optional[str] = None) -> None:
        """
        Listen for user input using speech-to-text and prompt the bot with the transcribed message.

        Args:
            name (Optional[str]): The name of the user sending the message. Defaults to None.
            init_time (Optional[int]): The time at which the listener was activated.
        """
        # Flag is set to True if a new user input is detected.
        input_detected = False

        # Listen for user input using speech-to-text
        for item in self._speech_recognition_service.recognize(init_time=init_time):
            # Do not send the message if it is empty.
            if sentence := item.value.display.strip():
                # Set the flag to True since a new user input was detected.
                input_detected = True

                # Send the transcribed message to the bot
                message_thread = threading.Thread(
                    target=self.send_message,
                    args=(
                        sentence,
                        ChatCompletionRoles.USER,
                        name,
                    ),
                    daemon=True,
                )
                self._thread_queue.add_task(message_thread, unskippable=True)

        if input_detected:
            self._thread_queue.add_task(
                threading.Thread(target=self.respond, kwargs={"init_time": init_time}, daemon=True)
            )
