import threading
import uuid
from logging import Logger
from operator import attrgetter
from typing import Callable, List, Optional, Tuple

from langchain.docstore.document import Document
from slack_bolt import BoltContext
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from sqlalchemy.exc import SQLAlchemyError
from weaviate.exceptions import WeaviateBaseError

import chatiq
from chatiq.block_builders import ChannelConfigurationBlockBuilder
from chatiq.channel_info_parser import ChannelInfoParser
from chatiq.document_loaders import MessageLoader
from chatiq.document_loaders.slack_link import SlackLinkLoader
from chatiq.document_loaders.unfurlink_link import UnfurlingLinkLoader
from chatiq.exceptions import TemperatureRangeError, TimezoneOffsetSelectError
from chatiq.models import SlackTeam
from chatiq.repositories import SlackTeamRepository
from chatiq.utils import subtract_documents
from chatiq.vectorstore import Vectorstore

from .base import BaseHandler


class MessageHandler(BaseHandler):
    """Event handler for the "message" Slack event.

    This handler is invoked when a message event occurs in a workspace.

    Attributes:
        chatiq: An instance of the ChatIQ class.
    """

    def __init__(self, chatiq: "chatiq.ChatIQ") -> None:
        """Initializes the MessageHandler with a ChatIQ instance.

        Args:
            chatiq (chatiq.ChatIQ): An instance of the ChatIQ class.
        """

        super().__init__(chatiq)

    def __call__(
        self, client: WebClient, context: BoltContext, body: dict, logger: Logger, say: Callable[..., None]
    ) -> None:
        """Handle a "message" event.

        Args:
            client (WebClient): The Slack WebClient to use for API calls.
            context (BoltContext): Event context with bot tokens for API calls.
            body (dict): The body of the Slack message event.
            logger (Logger): The logger to log debug and info messages.
            say (Callable[..., None]): The function to post messages.
        """

        subtype = body["event"].get("subtype")
        if subtype not in ("message_deleted", "message_changed", "channel_topic", "channel_purpose", "file_share", None):
            logger.debug(f"Unsupported message subtype is received: {subtype}")
            return

        thread = threading.Thread(target=self._handle, args=(client, context, body, logger, say))
        self.chatiq.add_thread(thread)
        thread.start()

    def _handle(self, client: WebClient, context: BoltContext, body: dict, logger: Logger, say: Callable[..., None]) -> None:
        """Actual handler function that processes the "message" event. This function is run in a separate thread.

        Args:
            client (WebClient): The Slack WebClient for making API calls.
            context (BoltContext): Event context with bot tokens for API calls.
            body (dict): The body of the Slack message event.
            logger (Logger): The logger to log debug and info messages.
            say (Callable[..., None]): The function to post messages.
        """

        subtype = body["event"].get("subtype")
        if subtype in ("channel_topic", "channel_purpose"):
            self._handle_channel_info_event(client, body, logger, say)
        elif body["event"].get("subtype") == "message_deleted":
            self._handle_message_deleted_event(body, logger)
        else:  # "message_changed", "file_share", None
            self._handle_message_created_and_changed_event(client, context, body, logger)

    def _handle_message_created_and_changed_event(
        self, client: WebClient, context: BoltContext, body: dict, logger: Logger
    ) -> None:
        """Handles created or changed messages.

        Args:
            client (WebClient): The Slack WebClient to use for API calls.
            context (BoltContext): Event context with bot tokens for API calls.
            body (dict): The event body.
            message (dict): The current message in the event body.
            logger (Logger): The logger to use.
        """

        if body["event"].get("subtype") == "message_changed":
            message = body["event"]["message"]
            previous_message = body["event"]["previous_message"]
        else:
            message = body["event"]
            previous_message = None

        try:
            logger.debug(f"Creating or changing message on team: {body['team_id']}")

            with self.chatiq.db.transaction() as session:
                repo = SlackTeamRepository(session)
                team = repo.get_or_create(body["team_id"], body["authorizations"][0]["user_id"])
                model, namespace_uuid = attrgetter("model", "namespace_uuid")(team)

            vectorstore = Vectorstore(self.chatiq.weaviate_client, body["team_id"])
            vectorstore.ensure_index()

            message_loader = MessageLoader(client, body, message, model)
            message_documents = message_loader.load()

            uuids = [uuid.uuid5(namespace_uuid, doc.metadata["ts"]) for doc in message_documents]  # type: ignore
            vectorstore.add_documents(message_documents, uuids=uuids)

            added_unfurling_link_documents, deleted_unfurling_link_documents = self._diff_unfurling_link_documents(
                body, message, previous_message, model
            )
            added_slack_link_documents, deleted_slack_link_documents = self._diff_slack_link_documents(
                body, message, previous_message, model
            )

            for document in added_unfurling_link_documents + added_slack_link_documents:
                # Iterating through the documents list and adding each document individually
                # to the vectorstore. While vectorstore.add_documents is capable of handling a list
                # of documents, a large batch of documents can cause a timeout error due to the
                # prolonged operation time. To prevent this, we add the documents one by one.
                vectorstore.add_documents([document])
            for document in deleted_unfurling_link_documents + deleted_slack_link_documents:
                vectorstore.delete_file_or_attachment(document.metadata["file_or_attachment_id"])

            logger.info(f"Added documents to vectorstore on team: {body['team_id']}")
        except SlackApiError as e:
            logger.error(f"SlackApiError: {e}")
        except WeaviateBaseError as e:
            logger.error(f"WeaviateBaseError: {e}")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemyError: {e}")
        except Exception as e:
            logger.error(f"Error: {e}")

    def _handle_message_deleted_event(self, body: dict, logger: Logger) -> None:
        """Handles deleting messages.

        Args:
            body (dict): The event body.
            logger (Logger): The logger to use.
        """

        try:
            logger.debug(f"Deletiing message on team: {body['team_id']}")

            vectorstore = Vectorstore(self.chatiq.weaviate_client, body["team_id"])
            vectorstore.ensure_index()

            vectorstore.delete_message(body["event"]["previous_message"]["ts"])

            logger.info(f"Deleted documents from vectorstore on team: {body['team_id']}")
        except WeaviateBaseError as e:
            logger.error(f"WeaviateBaseError: {e}")
        except Exception as e:
            logger.error(f"Error: {e}")

    def _handle_channel_info_event(self, client: WebClient, body: dict, logger: Logger, say: Callable[..., None]) -> None:
        """Handles channel topic and purpose change.

        Args:
            client (WebClient): The Slack WebClient for making API calls.
            body (dict): The event body.
            logger (Logger): The logger to use.
            say (Callable[..., None]): The function to post messages.
        """

        team_id = body["team_id"]
        channel_id = body["event"]["channel"]

        try:
            logger.debug(f"Changing channel info on team: {team_id}")

            topic, description = self._get_channel_info(client, channel_id)
            temperature, timezone_offset, context = ChannelInfoParser(topic, description).parse()
            blocks = ChannelConfigurationBlockBuilder.build_channel_configuration(temperature, timezone_offset, context)

            if blocks:
                say(text="Configuration is set for this channel.", blocks=blocks)

            logger.info(f"Changed channel info on team: {team_id}")
        except TemperatureRangeError as e:
            logger.error(f"TemperatureRangeError: {e}")
            say(
                "I'm sorry, something went wrong. "
                "Please ensure the AI temperature  :thermometer:  of this channel "
                f"is in range {SlackTeam.MIN_TEMPERATURE} - {SlackTeam.MAX_TEMPERATURE}.",
            )
        except TimezoneOffsetSelectError as e:
            logger.error(f"TimezoneOffsetSelectError: {e}")
            say(
                "I'm sorry, something went wrong. "
                "Please ensure the timezone offset  :round_pushpin:  of this channel "
                f"is one of {', '.join(SlackTeam.TIMEZONE_OFFSETS)}.",
            )
        except SlackApiError as e:
            logger.error(f"SlackApiError: {e}")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemyError: {e}")
        except Exception as e:
            logger.error(f"Error: {e}")

    def _diff_unfurling_link_documents(
        self, body: dict, message: dict, previous_message: Optional[dict], model: str
    ) -> Tuple[List[Document], List[Document]]:
        """This method loads documents from the current and previous messages, then returns added and deleted documents

        Args:
            body (dict): The body of the Slack message event.
            message (dict): The current message in the event body.
            previous_message (dict): The previous message in the event body.
            model (str): The model name.

        Returns:
            Tuple[List[Document], List[Document]]: A tuple containing two lists of documents. The first list contains
            documents that have been added, and the second list contains documents that have been deleted.
        """

        documents = self._load_unfurling_link_documents(body, message, model)
        if not previous_message:
            return documents, []

        previous_documents = self._load_unfurling_link_documents(body, previous_message, model)
        added_documents = subtract_documents(documents, previous_documents)
        deleted_documents = subtract_documents(previous_documents, documents)
        return added_documents, deleted_documents

    def _diff_slack_link_documents(
        self, body: dict, message: dict, previous_message: Optional[dict], model: str
    ) -> Tuple[List[Document], List[Document]]:
        """This method loads documents from the current and previous messages, then returns added and deleted documents

        Args:
            body (dict): The body of the Slack message event.
            message (dict): The current message in the event body.
            previous_message (dict): The previous message in the event body.
            model (str): The model name.

        Returns:
            Tuple[List[Document], List[Document]]: A tuple containing two lists of documents. The first list contains
            documents that have been added, and the second list contains documents that have been deleted.
        """

        documents = self._load_slack_link_documents(body, message, model)
        if not previous_message:
            return documents, []

        previous_documents = self._load_slack_link_documents(body, previous_message, model)
        added_documents = subtract_documents(documents, previous_documents)
        deleted_documents = subtract_documents(previous_documents, documents)
        return added_documents, deleted_documents

    def _load_unfurling_link_documents(self, body: dict, message: dict, model: str) -> List[Document]:
        """Loads documents from unfurling links in the message.

        This function iterates through the list of attachments in the message and loads the documents
        for those that contain valid unfurling links.

        Args:
            body (dict): The body of the Slack message event.
            message (dict): The message object from the event body.
            model (str): The model identifier.

        Returns:
            List[Document]: A list of Document objects loaded from the unfurling links.
        """

        if not message.get("attachments"):
            return []

        attachments = [
            attachment for attachment in message["attachments"] if UnfurlingLinkLoader.check_supported(attachment)
        ]

        documents = []
        for attachment in attachments:
            loader = UnfurlingLinkLoader(body, message, attachment, model)
            documents.extend(loader.load())
        return documents

    def _load_slack_link_documents(self, body: dict, message: dict, model: str) -> List[Document]:
        """Loads documents from Slack links in the message.

        This function iterates through the list of attachments in the message and loads the documents
        for those that contain valid Slack links.

        Args:
            body (dict): The body of the Slack message event.
            message (dict): The message object from the event body.
            model (str): The model identifier.

        Returns:
            List[Document]: A list of Document objects loaded from the Slack links.
        """

        if not message.get("attachments"):
            return []

        attachments = [attachment for attachment in message["attachments"] if SlackLinkLoader.check_supported(attachment)]

        documents = []
        for attachment in attachments:
            loader = SlackLinkLoader(body, message, attachment, model)
            documents.extend(loader.load())
        return documents

    def _get_channel_info(self, client: WebClient, channel_id: str) -> Tuple[str, str]:
        """Retrieves the topic and descriptionof a given Slack channel.

        Args:
            client (WebClient): The Slack WebClient for making API calls.
            channel_id (str): The ID of the Slack channel.

        Returns:
            Tuple[str, str]: The topic and description of the channel.
        """

        info = client.conversations_info(channel=channel_id)
        topic = info["channel"]["topic"]["value"]
        description = info["channel"]["purpose"]["value"]
        return topic, description
