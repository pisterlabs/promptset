import threading
from logging import Logger
from typing import Callable, List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from openai.error import InvalidRequestError, OpenAIError
from slack_sdk.errors import SlackApiError
from slack_sdk.web.client import WebClient
from sqlalchemy.exc import SQLAlchemyError
from weaviate.exceptions import WeaviateBaseError

import chatiq
from chatiq.channel_info_parser import ChannelInfoParser
from chatiq.chat_chain import ChatChain
from chatiq.exceptions import TemperatureRangeError, TimezoneOffsetSelectError
from chatiq.models import SlackTeam
from chatiq.repositories import SlackTeamRepository
from chatiq.text_processor import TextProcessor
from chatiq.vectorstore import Vectorstore

from .base import BaseHandler


class AppMentionHandler(BaseHandler):
    """Event handler for the "app_mention" Slack event.

    This handler is used to define the behavior of the AI assistant when it is mentioned in a Slack message.

    Attributes:
        chatiq: An instance of the ChatIQ class.
    """

    def __init__(self, chatiq: "chatiq.ChatIQ") -> None:
        """Initializes the AppMentionHandler with a ChatIQ instance.

        Args:
            chatiq (chatiq.ChatIQ): An instance of the ChatIQ class.
        """

        super().__init__(chatiq)

    def __call__(self, client: WebClient, body: dict, logger: Logger, say: Callable[..., None]) -> None:
        """Event handler for the “app_mention" Slack event.

        When the bot is mentioned in a message, this method is called to handle the event. To prevent the
        handler from taking too long and causing Slack to send duplicate events, a new thread is started to
        handle the event's processing.

        Args:
            client (WebClient): The Slack WebClient for making API calls.
            body (dict): The body of the Slack message event.
            logger (Logger): The logger to log debug and info messages.
            say (Callable[..., None]): The function to post messages.
        """

        # Skip event handling when message is edited
        if "edited" in body["event"]:
            return

        thread = threading.Thread(target=self._handle, args=(client, body, logger, say))
        self.chatiq.add_thread(thread)
        thread.start()

    def _handle(self, client: WebClient, body: dict, logger: Logger, say: Callable[..., None]) -> None:
        """Actual handler function that processes the "app_mention" event. This function is run in a separate thread.

        Args:
            client (WebClient): The Slack WebClient for making API calls.
            body (dict): The body of the Slack message event.
            logger (Logger): The logger to log debug and info messages.
            say (Callable[..., None]): The function to post messages.
        """

        bot_id = body["authorizations"][0]["user_id"]
        team_id = body["team_id"]
        channel_id = body["event"]["channel"]
        thread_ts = body["event"].get("thread_ts", body["event"]["ts"])

        try:
            logger.debug(f"App mentioned on team: {team_id}")

            topic, description, is_private = self._get_channel_info(client, channel_id)
            temperature, timezone_offset, context = ChannelInfoParser(topic, description).parse()

            with self.chatiq.db.transaction() as session:
                repo = SlackTeamRepository(session)
                team = repo.get_or_create(team_id, bot_id)
                model = team.model
                temperature = temperature if temperature is not None else team.temperature
                timezone_offset = timezone_offset or team.timezone_offset
                context = context or team.context

            token_length = TextProcessor.get_token_length_for_model(model)  # type: ignore
            chat = ChatOpenAI(model_name=model, temperature=temperature)

            memory = ConversationTokenBufferMemory(
                llm=chat,
                memory_key="chat_history",
                output_key="output",
                input_key="input",
                return_messages=True,
                max_token_limit=token_length,  # type: ignore
            )

            vectorstore = Vectorstore(self.chatiq.weaviate_client, team_id)
            vectorstore.ensure_index()
            retriever = vectorstore.as_retriever(is_private, channel_id, thread_ts)

            chain = ChatChain(chat, memory, retriever, bot_id, channel_id, context, timezone_offset)

            messages = self._get_messages(client, channel_id, thread_ts)
            for message in messages[:-1]:
                if message["user"] == bot_id:
                    chain.add_memory_ai_message(message)
                else:
                    chain.add_memory_user_message(message)

            say(chain.run(messages[-1]), thread_ts=thread_ts)

            logger.info(f"Replies to app mention on team: {team_id}")
        except TemperatureRangeError as e:
            logger.error(f"TemperatureRangeError: {e}")
            say(
                "I'm sorry, something went wrong. "
                "Please ensure the AI temperature  :thermometer:  of this channel "
                f"is in range {SlackTeam.MIN_TEMPERATURE} - {SlackTeam.MAX_TEMPERATURE}.",
                thread_ts=thread_ts,
            )
        except TimezoneOffsetSelectError as e:
            logger.error(f"TimezoneOffsetSelectError: {e}")
            say(
                "I'm sorry, something went wrong. "
                "Please ensure the timezone offset  :round_pushpin:  of this channel "
                f"is one of {', '.join(SlackTeam.TIMEZONE_OFFSETS)}.",
                thread_ts=thread_ts,
            )
        except InvalidRequestError as e:
            logger.error(f"InvalidRequestError: {e}")
            say(
                "I'm sorry, something went wrong. Your message might be too large. "
                "Plese you try reducing the size and send it again.",
                thread_ts=thread_ts,
            )
        except OpenAIError as e:
            logger.error(f"OpenAIError: {e}")
            say(
                "I'm sorry, something went wrong. "
                "Please ensure that your OpenAI API key is valid and you have enough quota.",
                thread_ts=thread_ts,
            )
        except SlackApiError as e:
            logger.error(f"SlackApiError: {e}")
            say("I'm sorry, something went wrong. Please ensure the bot has the correct permissions.", thread_ts=thread_ts)
        except WeaviateBaseError as e:
            logger.error(f"WeaviateBaseError: {e}")
            say("I'm sorry, something went wrong.", thread_ts=thread_ts)
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemyError: {e}")
            say("I'm sorry, something went wrong.", thread_ts=thread_ts)
        except Exception as e:
            logger.error(f"Error: {e}")
            say("I'm sorry, something went wrong.", thread_ts=thread_ts)

    def _get_channel_info(self, client: WebClient, channel_id: str) -> Tuple[str, str, bool]:
        """Retrieves the topic, description and privacy status of a given Slack channel.

        Args:
            client (WebClient): The Slack WebClient for making API calls.
            channel_id (str): The ID of the Slack channel.

        Returns:
            Tuple[str, str, bool]: The topic, description and the privacy status of the channel.
        """

        info = client.conversations_info(channel=channel_id)
        topic = info["channel"]["topic"]["value"]
        description = info["channel"]["purpose"]["value"]
        is_private = info["channel"]["is_private"]
        return topic, description, is_private

    def _get_messages(self, client: WebClient, channel_id: str, thread_ts: str) -> List[dict]:
        """Retrieves the previous messages in the thread where the bot was mentioned.

        Args:
            client (WebClient): The Slack WebClient for making API calls.
            channel_id (str): The ID of the Slack channel.
            thread_ts (str): The ID of the thread.

        Returns:
            List[dict]: The previous messages in the thread.
        """

        conversation_replies = client.conversations_replies(channel=channel_id, ts=thread_ts)
        return conversation_replies["messages"]
