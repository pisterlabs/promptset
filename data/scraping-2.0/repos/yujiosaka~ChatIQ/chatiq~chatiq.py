import logging
import signal
import sys
import threading
from typing import Any, List, Optional, Set

import openai
from slack_bolt import App
from slack_bolt.oauth.oauth_settings import OAuthSettings
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler
from weaviate import Client
from weaviate.exceptions import WeaviateBaseError

from chatiq.database import Database
from chatiq.exceptions import SettingsValidationError
from chatiq.handlers import (
    AppHomeOpenedHandler,
    AppMentionHandler,
    AppUninstalledHandler,
    ChannelDeletedHandler,
    ContextSaveHandler,
    FileDeletedHandler,
    FileSharedHandler,
    MessageHandler,
    ModelSelectHandler,
    TemperatureSelectHandler,
    TimezoneOffsetSelectHandler,
)
from chatiq.settings import Settings


class ChatIQ:
    """Slack bot class that uses the OpenAI GPT model to provide responses to mentions and messages in a Slack workspace.

    Attributes:
        bolt_app (App): An instance of the Bolt App. This instance is configured with the event handlers
        and settings for the bot, and it's intended to be public so that you can run the app using this instance.
    """

    BOT_TOKEN_SCOPES: List[str] = [
        "app_mentions:read",
        "channels:history",
        "channels:read",
        "chat:write",
        "groups:history",
        "groups:read",
        "mpim:history",
        "mpim:read",
        "files:read",
    ]
    COMMON_REQUIRED_SETTINGS: Set[str] = {"openai_api_key", "postgres_url", "weaviate_url"}
    REQUIRED_SETTINGS_WITHOUT_BOLT_APP: Set[str] = {"slack_client_id", "slack_client_secret", "slack_signing_secret"}

    def __init__(
        self,
        bolt_app: Optional[App] = None,
        slack_client_id: Optional[str] = None,
        slack_client_secret: Optional[str] = None,
        slack_signing_secret: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        postgres_url: Optional[str] = None,
        weaviate_url: Optional[str] = None,
        rate_limit_retry: Optional[bool] = False,
    ) -> None:
        """Initialize the ChatIQ bot with the provided settings.

        Args:
            bolt_app (App, optional): An existing Slack Bolt app to add the ChatIQ functionality.
                If None, a new Bolt app is created. Defaults to None.
            slack_client_id (str, optional): The client ID for the Slack app. Defaults to None.
            slack_client_secret (str, optional): The client secret for the Slack app. Defaults to None.
            slack_signing_secret (str, optional): The signing secret for the Slack app. Defaults to None.
            openai_api_key (str, optional): The API key for OpenAI. Defaults to None.
            postgres_url (str, optional): The URL for the Postgres database. Defaults to None.
            weaviate_url (str, optional): The URL for the Weaviate instance. Defaults to None.
            rate_limit_retry (bool, optional): Whether to enable the rate limit retry handler. Defaults to False.

        Raises:
            SettingsValidationError: If any of the required settings are missing.
            WeaviateBaseError: If there is an issue with connecting to the Weaviate instance.
            SQLAlchemyError: If there is an issue creating the tables in the database.
        """

        logging.basicConfig(level=getattr(logging, Settings.LOG_LEVEL))
        self.logger = logging.getLogger(__name__)

        self.slack_client_id = slack_client_id or Settings.SLACK_CLIENT_ID
        self.slack_client_secret = slack_client_secret or Settings.SLACK_CLIENT_SECRET
        self.slack_signing_secret = slack_signing_secret or Settings.SLACK_SIGNING_SECRET
        self.openai_api_key = openai_api_key or Settings.OPENAI_API_KEY
        self.postgres_url = postgres_url or Settings.POSTGRES_URL
        self.weaviate_url = weaviate_url or Settings.WEAVIATE_URL
        self._validate_settings(bool(bolt_app))

        openai.api_key = self.openai_api_key

        self.weaviate_client = self._initialize_weaviate_client()
        self.db = self._initialize_database()

        self.bolt_app = bolt_app or self._initialize_app()
        if rate_limit_retry:
            self.bolt_app.client.retry_handlers.append(RateLimitErrorRetryHandler())

        self.threads = []
        self.thread_lock = threading.Lock()

    def listen(self) -> None:
        """Start listening for Slack events.

        After calling this method, the bot will start listening for events like mentions,
        messages, app installations, and uninstallations. Handlers for these events should be
        defined before calling this method. In addition, this method also sets up a handler for
        SIGTERM to allow for graceful shutdown of the bot.
        """

        signal.signal(signal.SIGTERM, self._exit_gracefully)

        self.bolt_app.event("message")(MessageHandler(self))
        self.bolt_app.event("file_shared")(FileSharedHandler(self))
        self.bolt_app.event("file_deleted")(FileDeletedHandler(self))
        self.bolt_app.event("app_mention")(AppMentionHandler(self))
        self.bolt_app.event("app_uninstalled")(AppUninstalledHandler(self))
        self.bolt_app.event("app_home_opened")(AppHomeOpenedHandler(self))
        self.bolt_app.event("channel_deleted")(ChannelDeletedHandler(self))
        self.bolt_app.event("group_deleted")(ChannelDeletedHandler(self))

        self.bolt_app.action("model_select")(ModelSelectHandler(self))
        self.bolt_app.action("temperature_select")(TemperatureSelectHandler(self))
        self.bolt_app.action("timezone_offset_select")(TimezoneOffsetSelectHandler(self))
        self.bolt_app.action("context_save")(ContextSaveHandler(self))

    def add_thread(self, thread: threading.Thread):
        """Adds a thread to the list after cleaning up finished threads.

        Args:
            thread (threading.Thread): The thread to be added.
        """

        with self.thread_lock:
            self._clean_threads()
            self.threads.append(thread)

    def _clean_threads(self):
        """Removes finished threads from the list."""

        self.threads = [thread for thread in self.threads if thread.is_alive()]

    def _exit_gracefully(self, signum: int, frame: Any):
        """Handles SIGTERM by waiting for all threads to complete before exiting.

        Args:
            signum (int): The signal number.
            frame: The current stack frame (unused in this method but required by the signal module).
        """

        for thread in self.threads:
            thread.join()
        sys.exit(0)

    def _validate_settings(self, with_bolt_app: bool) -> None:
        """Validates the provided settings.

        This method checks whether all required settings have been provided.
        If any required setting is missing, it raises a  SettingsValidationError
        with a message specifying which settings are missing.

        Args:
            with_bolt_app (bool): Indicates whether a bolt app has been provided.
            If True, the method does not require Slack app settings to be configured.

        Raises:
            SettingsValidationError: If any of the required settings are missing.
        """

        if with_bolt_app:
            required_settings = self.COMMON_REQUIRED_SETTINGS
        else:
            required_settings = self.COMMON_REQUIRED_SETTINGS | self.REQUIRED_SETTINGS_WITHOUT_BOLT_APP

        missing_settings = [name for name, value in vars(self).items() if name in required_settings and not value]

        if with_bolt_app and any(setting in vars(self) for setting in self.REQUIRED_SETTINGS_WITHOUT_BOLT_APP):
            self.logger.warning(
                "Unnecessary Slack app settings are configured. These settings are not required when a bolt app is provided."
            )

        if missing_settings:
            raise SettingsValidationError(
                f"Missing settings: {', '.join(missing_settings)}. "
                f"Please provide them in environment variables or pass them as arguments."
            )

    def _initialize_weaviate_client(self):
        """Initialize the Weaviate client with the provided Weaviate URL.

        This method is responsible for creating an instance of the Weaviate client and ensuring
        that it can connect to the Weaviate instance at the provided URL. It's separated from the
        constructor to allow for dependency injection during testing.

        Returns:
            Client: The initialized Weaviate client.

        Raises:
            WeaviateBaseError: If there is an issue with connecting to the Weaviate instance.
        """

        try:
            weaviate_client = Client(self.weaviate_url)
        except Exception as e:
            error_message = f"Failed to connect to Weaviate. Error: {e}"
            self.logger.error(error_message)
            raise WeaviateBaseError(error_message)

        return weaviate_client

    def _initialize_database(self):
        db = Database(self.postgres_url, self.slack_client_id)
        db.setup()
        return db

    def _initialize_app(self) -> App:
        """Initializes the Slack Bolt App instance with OAuth settings configured.

        The OAuth settings are crucial for enabling the app to be installed
        across multiple workspaces. It sets up an installation store and
        a state store which are responsible for storing installation data
        (tokens, installation IDs, enterprise IDs, user IDs) and state
        parameters for OAuth flow respectively.

        Returns:
            App: An instance of Slack's Bolt App with OAuth configured.
        """

        oauth_settings = OAuthSettings(
            client_id=self.slack_client_id,
            client_secret=self.slack_client_secret,
            scopes=self.BOT_TOKEN_SCOPES,
            installation_store=self.db.installation_store,
            state_store=self.db.state_store,
        )
        app = App(
            logger=self.logger,
            signing_secret=self.slack_signing_secret,
            oauth_settings=oauth_settings,
            request_verification_enabled=False,
            process_before_response=True,
        )

        return app
