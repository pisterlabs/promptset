"""
Functionality for making inquiries to chatbots
"""
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, NewType, TypedDict
from operator import itemgetter

import backoff
import openai
import tiktoken
from opentelemetry.trace.span import Span
from openai.error import Timeout, APIConnectionError, ServiceUnavailableError, APIError

from .agents import Agent
from .errors import (
    RecoverableError,
    RateLimitError,
    QuotaExceededError,
    TemporaryAPIError,
    NoCompletionResultError,
    TooManyTokensError,
)
from .personas import Persona, PERSONAS

LOGGER = logging.getLogger("Brain Conductor")


def num_tokens_from_string(string: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Returns the number of tokens in a text string.
    There's no issue with performance from measurements. Typical length of time to
    calculate is less than 3 milliseconds
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    if model == "gpt-3.5-turbo" and num_tokens > 4000:
        raise TooManyTokensError()
    if model == "gpt-4" and num_tokens > 8000:
        raise TooManyTokensError()
    return num_tokens


Encoding = NewType("Encoding", str)
MimeType = NewType("MimeType", str)


class InquiryResponseDataType(Enum):
    """Data types"""

    IMAGE = "image"


@dataclass
class InquiryResponseData:
    """Response Data from an inquiry"""

    data: str
    type: InquiryResponseDataType
    encoding: Encoding
    mime_type: MimeType


@dataclass
class InquiryResponse:
    """Response form an inquiry"""

    message: str
    data: list[InquiryResponseData] = field(default_factory=list)


class InquiryManager:
    """
    Manager of chatbot inquiry logic in a Quart app
    """

    def __init__(
        self,
        openai_api_key: str,
        chat_model: str,
        text_model: str,
        personas: list[Persona],
        agents: Sequence[Agent],
        recent_items: int = 10,
    ) -> None:
        openai.api_key = openai_api_key
        self._chat_model = chat_model
        self._text_model = text_model
        self._personas: list[Persona] = personas
        self._recent_items = recent_items
        self._agents = agents

    def __enter__(self):
        return InquiryContextManager(
            self._chat_model,
            self._text_model,
            self._personas,
            self._recent_items,
            self._agents,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class InquiryContextManager:
    """
    Context manager for the Inquire manager to manage websocket session
    level inquiries/
    """

    def __init__(
        self,
        chat_model: str,
        text_model: str,
        personas: list[Persona],
        recent_items: int,
        agents: Sequence[Agent],
    ) -> None:
        self._chat_model = chat_model
        self._text_model = text_model
        self._personas: list[Persona] = personas
        self._recent_items = recent_items
        self._agents = agents

        self._history: list[tuple[Persona | None, str]] = []
        self._tokens = 0
        self.__persona_roles_text = None
        self.__persona_topics = None
        self.__persona_names = None
        self.__persona_full_names = None

    @staticmethod
    def __get_intersecting_topics_from_persona(persona: Persona, topics: list[str]):
        return [topic for topic in persona.topics if topic.strip().lower() in topics]

    def build_persona_list(
        self, appropriate_topics
    ) -> tuple[Persona | None, list[Persona]]:
        """
        Will add up all the topic scores across all relevant topics to choose the most
        appropriate topic leader.
        In the future, topic leader could also be weighted by previous conversation
        history
        :param appropriate_topics: The relevant topics to the user input
        :return: The primary persona and a list of secondary personas
        """

        class PersonaTopicScoresDict(TypedDict):
            persona: Persona
            value: int

        persona_topic_scores: list[PersonaTopicScoresDict] = list()
        for persona in self._personas:
            score = 0

            # Calculate a score based on intersecting topics and their value
            intersecting_topics = self.__get_intersecting_topics_from_persona(
                persona, appropriate_topics
            )
            for topic in intersecting_topics:
                score += persona.topics[topic]

            # If a score has been calculated, append that persona and its score,
            # so it can be elected to be a responder.
            if score:
                persona_topic_scores.append({"persona": persona, "value": score})

        persona_list: list[Persona] = list()
        if persona_topic_scores:
            # Sort all the scores to put the highest value in the front
            persona_topic_scores = sorted(persona_topic_scores, key=itemgetter("value"))

            # Determine the max score in order to randomize who the primary responder will be
            max_score = max([score["value"] for score in persona_topic_scores])
            # Those who have a max score and agents will take priority
            max_agents = [
                score
                for score in persona_topic_scores
                if score["value"] == max_score and score["persona"].agent
            ]
            # Randomize those who have max and priority and designate a primary
            if max_agents:
                random.shuffle(max_agents)
                primary_obj = max_agents.pop()
            else:
                # If no agents were found at max score randomize everyone with that value
                max_personas = [
                    score
                    for score in persona_topic_scores
                    if score["value"] == max_score
                ]
                random.shuffle(max_personas)
                primary_obj = max_personas.pop()
            persona_topic_scores.remove(primary_obj)
            primary = primary_obj["persona"]

            # Finally calculate a list of secondaries. We are limiting the
            # amount of secondaries to 2. Then shuffling that lists order as well.
            while len(persona_list) < 2 and persona_topic_scores:
                persona_list.append(persona_topic_scores.pop()["persona"])
            random.shuffle(persona_list)
        else:
            default_personas = [
                persona for persona in self._personas if persona.is_default_persona
            ]
            primary = random.choice(default_personas)
            persona_list = [
                persona for persona in default_personas if not persona == primary
            ]

        random.shuffle(persona_list)
        return primary, persona_list

    async def identify_personas(
        self, inquiry: str, request_span: Span
    ) -> tuple[Persona | None, list[Persona]]:
        """
        Identify a primary persona and zero or more secondary personas
        to which you wish to send the provided inquiry.
        :param inquiry: Inquiry to send
        :param request_span: Tracing span for tracing and debugging
        :return: A primary persona and zero or more secondary personas. If no
        personas could be identified, the primary persona is null.
        """
        target_name = False
        for name in self._persona_names:
            if name in inquiry.lower():
                target_name = True
                break
        if target_name:
            # Query LLM to see who should be addressed
            prompt = f"""
            Taking into consideration the user input, which person of the following
            should respond: {self._persona_full_names}
            EXAMPLES:
            Input: Larry, what is your favorite color?
            A: larry

            Input: I'd like to hear a joke from Comical Chris
            A: chris

            Input: Evaluate the square root of the function
            A: None

            Input: Shut up Harry
            A: None

            Input: Tell me how to build a business @erin
            A: erin

            Input: What is your opinion Narrative Nick and Gabby?
            A: nick, gabby

            Input: {inquiry}
            A:
            """
            try:
                request_span.set_attribute("personas.prompt", prompt)
                num_tokens_from_string(prompt)
                LOGGER.debug(f"Sending completion request with prompt: {prompt}")
                response_names: str = await self._openai_text_complete(prompt)
                request_span.set_attribute("personas.response", response_names)
                LOGGER.debug(f"Completion request returned: {response_names}")
            except openai.error.RateLimitError as e:
                raise RateLimitError(e)
            except TooManyTokensError:
                raise

            appropriate_personas = [
                name.strip().lower() for name in response_names.split(",")
            ]
            personas = [
                item.strip().lower()
                for item in appropriate_personas
                if item.strip().lower() in self._persona_names
            ]
            if len(personas) >= 1:
                default_personas = list()
                for persona in self._personas:
                    if persona.prompt_name.lower() in personas:
                        default_personas.append(persona)
                if len(default_personas) == 1:
                    return default_personas[0], []
                elif len(default_personas) > 1:
                    import random

                    primary = random.choice(default_personas)
                    persona_list = [
                        persona
                        for persona in default_personas
                        if not persona == primary
                    ]
                    random.shuffle(persona_list)
                    return primary, persona_list

        interactions = ""
        for persona, text in self._recent_history[-1:]:
            role = persona.role if persona else "user"
            interactions += f"\n{role}: {text}"

        interactions = interactions if interactions else "None"
        prompt = f"""
            Taking into consideration the previous interactions, which of the following
            topics does the question fall under?
            INTERACTIONS: {interactions}
            TOPICS: {self._persona_topics}
            EXAMPLES:

            Q: How do I run?
            A:

            Q: Why is the sky blue?
            A: Science, Philosophy

            Q: How do you cook an egg?
            A: Food

            Q: How many times can a salamander regrow its tail?
            A: Animals, Science

            Q: Which Madden sports game was the most successful?
            A: Sports, Business, Gaming

            QUESTION: {inquiry}
            ANSWER: 
            """  # noqa: W291

        try:
            request_span.set_attribute("personas.prompt", prompt)
            num_tokens_from_string(prompt)
            LOGGER.debug(f"Sending completion request with prompt: {prompt}")
            response: str = await self._openai_text_complete(prompt)
            request_span.set_attribute("personas.response", response)
            LOGGER.debug(f"Completion request returned: {response}")
        except openai.error.RateLimitError as e:
            raise RateLimitError(e)
        except TooManyTokensError:
            raise

        appropriate_topics = [topic.strip().lower() for topic in response.split(",")]
        return self.build_persona_list(appropriate_topics)

    async def inquire(self, persona: Persona, inquiry: str) -> InquiryResponse:
        """
        Make in inquiry a chatbot persona
        :param persona: Persona to which the inquiry is destined
        :param inquiry: Question to send to the persona
        :return: Response from the persona
        """
        message = {
            "role": "user",
            "content": inquiry,
        }
        return await self.chat_complete(persona, message, None)

    async def comment_on_history(self, persona: Persona) -> InquiryResponse:
        """
        Request a chatbot persona to comment on the chat history
        :param persona: Persona from which you wish to receive a comment
        :return: The persona's comment
        """
        message = None
        instruction = (
            f"In at most 3 sentences, respond to the conversation "
            f"as {persona.name} about subject matter that you are associated with. "
            f"Pretend this is an improv session and you have the next line. "
            f"Since it's a conversation you don't need to include your own name or "
            f"put the response in quotations. Make a strong effort "
            f"to keep on topic by viewing the user's previous messages, and also "
            f"take the other experts responses into secondary account."
        )
        return await self.chat_complete(persona, message, instruction)

    async def chat_complete(
        self, persona: Persona, message: dict[str, str] | None, instruction: str | None
    ) -> InquiryResponse:
        """
        Request a chat completion from a chatbot persona
        :param persona: Persona from which you wish to complete the chat
        :param message: Message with the sender and text to send the persona
        :param instruction: Specific instruction on how to complete the chat
        :return: Response from the chatbot persona
        """
        if not instruction:
            instruction = (
                f"Respond in the voice of {persona.name}, with the knowledge your "
                f"persona would have about the subject matter. Make a strong effort "
                f"to keep on topic by viewing the user's previous messages."
            )
        messages = [
            {
                "role": "system",
                "content": f"You are not an AI, but the persona of a "
                f"caricatured and eccentric "
                f"{persona.role} named {persona.name}. {persona.description}"
                f"{instruction}. Never refer to yourself in the third person, "
                "or mention your name at all.",
            },
        ]
        for historical_persona, text in self._recent_history:
            if historical_persona:
                historical_message = {
                    "role": "assistant",
                    "name": historical_persona.prompt_name,
                    "content": text,
                }
            else:
                historical_message = {
                    "role": "user",
                    "content": text,
                }
            messages.append(historical_message)

        if message:
            messages.append(message)
            self._history.append((None, message["content"]))

        while len(messages) >= 3:
            try:
                num_tokens_from_string(repr(messages))
                break
            except TooManyTokensError:
                # Try to get rid of message history to shorten the tokens
                messages.pop(1)
                if len(messages) < 3:
                    # No longer has the inquiry and the persona
                    raise

        LOGGER.debug(f"Sending chat completion request with messages: {messages}")
        data_items: list[InquiryResponseData] = []
        if persona.agent:
            agent = self._get_agent(persona.agent)
            agent_response = await agent.process_messages(messages)
            for image in agent_response.images:
                data_items.append(
                    InquiryResponseData(
                        data=image,
                        type=InquiryResponseDataType.IMAGE,
                        encoding=Encoding("base64"),
                        mime_type=MimeType("image/jpeg"),
                    )
                )
            response_message = agent_response.response
        else:
            response_message = await self._openai_chat_complete(messages)

        LOGGER.debug(f"Sending chat completion request returned {response_message}")
        response_message = self.finalize_response(response_message, messages)
        self._history.append((persona, response_message))
        response = InquiryResponse(message=response_message, data=data_items)
        return response

    def finalize_response(self, response: str, messages: list[dict[str, str]]) -> str:
        """
        Finalize the response before returning to the user. This contains logic
        to ensure the response make sense to the inquirer and is within the realm
        of conversation we wish to expose.
        :param response: Chatbot persona's response
        :param messages: Messages used for context
        :return: Finalized chatbot persona response
        """
        triggers = [
            "language model",
            "real person",
            "fictional character",
            "openai",
            "assistant ai",
            "as an ai",
            "talking ai",
            "as a chatbot",
            "as ai",
            "ai assistantx",
        ]
        if len([s for s in triggers if s in response.lower()]) > 0:
            revision = [
                messages[0],
                {
                    "role": "system",
                    "content": f"``` {response}``` "
                    "Rephrase the content included in backticks so that it "
                    "sounds like it's coming from your described "
                    "persona as if you were a real person",
                },
            ]
            try:
                chat_completion = openai.ChatCompletion.create(
                    model=self._chat_model, messages=revision
                )
            except openai.error.RateLimitError as e:
                raise RateLimitError(e)
            response = chat_completion.choices[0].message.content
        return response

    @property
    def tokens(self):
        """
        Tokens property
        :return: Total number of tokens utilized
        """
        return self._tokens

    @property
    def _persona_roles_text(self):
        if not self.__persona_roles_text:
            self.__persona_roles_text = ", ".join(
                [persona.role for persona in self._personas]
            )
        return self.__persona_roles_text

    @property
    def _persona_topics(self):
        if not self.__persona_topics:
            all_topics = set()
            for persona in self._personas:
                topics = [topic.strip().lower() for topic in persona.topics]
                all_topics = all_topics.union(set(topics))
            self.__persona_topics = all_topics
        return self.__persona_topics

    @property
    def _persona_names(self):
        if not self.__persona_names:
            names = list()
            for persona in self._personas:
                names.append(persona.prompt_name.strip().lower())
            self.__persona_names = names
        return self.__persona_names

    @property
    def _persona_full_names(self):
        if not self.__persona_full_names:
            names = list()
            for persona in self._personas:
                names.append(persona.name)
            self.__persona_full_names = names
        return self.__persona_full_names

    @property
    def _recent_history(self):
        return self._history[-self._recent_items :]  # noqa: E203

    @backoff.on_exception(backoff.expo, RecoverableError)
    async def _openai_chat_complete(self, messages: list[dict[str, str]]):
        try:
            chat_completion = await openai.ChatCompletion.acreate(
                model=self._chat_model, messages=messages
            )
            if not chat_completion:
                raise NoCompletionResultError(
                    "No chat completion result returned from OpenAI"
                )

            self._tokens += chat_completion.usage.total_tokens
            response = chat_completion.choices[0].message.content
            return response
        except openai.error.RateLimitError as e:
            if "quota" in e.user_message:
                raise QuotaExceededError(e)
            elif "overloaded" in e.user_message:
                raise TemporaryAPIError(e)
            else:
                raise RateLimitError(e)
        except (Timeout, APIConnectionError, ServiceUnavailableError) as e:
            raise TemporaryAPIError(e)
        except APIError as e:
            if 500 <= e.code < 600:
                raise TemporaryAPIError(e)
            raise

    @backoff.on_exception(backoff.expo, RecoverableError)
    async def _openai_text_complete(self, text) -> str:
        try:
            completion = await openai.Completion.acreate(
                model=self._text_model, prompt=text
            )
            if not completion:
                raise NoCompletionResultError(
                    "No text completion result returned from OpenAI"
                )
            self._tokens += completion.usage.total_tokens
            response = completion.choices[0].text
            return response
        except openai.error.RateLimitError as e:
            if "quota" in e.user_message:
                raise QuotaExceededError(e)
            elif "overloaded" in e.user_message:
                raise TemporaryAPIError(e)
            else:
                raise RateLimitError(e)
        except (Timeout, APIConnectionError, ServiceUnavailableError) as e:
            raise TemporaryAPIError(e)
        except APIError as e:
            if 500 <= e.code < 600:
                raise TemporaryAPIError(e)
            raise

    def prepend_history(self, persona_name: str, text: str):
        """
        :param persona_name: Name of agent
        :param text: Message sent by agent
        """
        if persona_name:
            matches = [persona for persona in PERSONAS if persona.name == persona_name]
            persona = matches[0] if matches else None
        else:
            persona = None
        self._history.insert(0, (persona, text))

    def _get_agent(self, agent_type):
        for agent in self._agents:
            if isinstance(agent, agent_type):
                return agent
        raise ValueError(f"No agent for type {agent_type}")
