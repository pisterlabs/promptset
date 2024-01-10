import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Union

from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from .SimpleThrottle import SimpleThrottle


class AsyncStreamingSlackCallbackHandler(AsyncCallbackHandler):
    """Async callback handler for streaming to Slack. Only works with LLMs that support streaming."""

    def __init__(self, client: WebClient):
        self.client = client
        self.channel_id = None
        self.thread_ts = None
        self.update_delay = 0.1  # Set the desired delay in seconds
        self.update_throttle = SimpleThrottle(self._update_message_in_slack, self.update_delay)

    async def start_new_response(self, channel_id, thread_ts):
        self.current_message = ""
        self.message_ts = None
        self.channel_id = channel_id
        self.thread_ts = thread_ts

    async def _update_message_in_slack(self):
        try:
            await self.client.chat_update(
                channel=self.channel_id, ts=self.message_ts, text=self.current_message
            )
        except SlackApiError as e:
            print(f"Error updating message: {e}")

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.current_message += token
        await self.update_throttle.call()


    async def handle_llm_error(self, e: Exception) -> None:
        """Post error message to channel with provided channel_id and thread_ts."""
        try:
            logger.error(f"Got LLM Error. Will post about it: {e}")
            await self.client.chat_postMessage(text=str(e), channel=self.channel_id, thread_ts=self.thread_ts)
        except Exception as e:
            logger.error(f"Error posting exception message: {e}")

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        try:
            if self.channel_id is None:
                raise Exception("channel_id is None")
            # Send an empty response and get the timestamp
            post_response = await self.client.chat_postMessage(text="...", channel=self.channel_id, thread_ts=self.thread_ts)
            self.message_ts: str = post_response["ts"]
        except Exception as e:
            await self.handle_llm_error(e)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        try:
            await self.update_throttle.call_and_wait()
            # Make sure it got the last one:
            await self.start_new_response(self.channel_id, self.thread_ts)
        except Exception as e:
            await self.handle_llm_error(e)

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        await self.handle_llm_error(error)

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    async def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    async def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    async def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        print("Got text!", text)

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
