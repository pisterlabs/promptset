import asyncio
import re
import warnings
from textwrap import dedent, shorten
from typing import AsyncIterator, Literal, overload

import openai
import yaml
from discord import Attachment, Embed, Message, MessageType, Thread
from discord.abc import Messageable
from loguru import logger
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from more_itertools import constrained_batches, first, locate, split_at

from chatbot.modules.chat.helpers import num_tokens_from_messages
from chatbot.modules.chat.models import (
    CHAT_MODELS,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageType,
    ChatSessionOptions,
)
from chatbot.settings import AppSecrets
from chatbot.utils.config import load_settings
from chatbot.utils.discord.color import Color2
from chatbot.utils.discord.embed import Embed2
from chatbot.utils.discord.file import discord_open
from chatbot.utils.discord.markdown import divide_text
from chatbot.utils.discord.messageable import send_message
from chatbot.utils.discord.typing import OutgoingMessage
from chatbot.utils.errors import (
    is_system_message,
    report_error,
    report_warnings,
    system_message,
)

MAX_MESSAGE_LENGTH = 1996


class ChatSession:
    SECRETS = load_settings(AppSecrets)

    def __init__(
        self,
        assistant: str,
        options: ChatSessionOptions,
    ):
        self.assistant = assistant

        self.options = options
        self.messages: list[ChatMessage] = []

        self.editing = asyncio.Lock()

        self.token_usage: int = 0
        self.token_estimate: int = 0
        self.estimate_token_usage(removed=[], added=self.options.request.messages)

    @property
    def all_messages(self) -> list[ChatMessage]:
        """Create a list of all messages including messages from the preset."""
        return [
            *self.options.request.messages,
            *self.messages,
        ]

    @property
    def conversational_messages(self) -> list[ChatMessage]:
        """Create a list of all messages excluding system messages."""
        return [*filter(lambda m: m.role != "system", self.all_messages)]

    @property
    def usage_description(self) -> str:
        if self.token_usage != self.token_estimate:
            return f"{self.token_count_upper_bound} (estimated)"
        return f"{self.token_usage}"

    @property
    def token_count_upper_bound(self) -> int:
        return max(self.token_estimate, self.token_usage)

    @property
    def system_message(self) -> str | None:
        if not self.options.request.messages:
            return None
        message = self.options.request.messages[0]
        if message.role != "system":
            return
        return message.content

    def estimate_token_usage(
        self,
        *,
        removed: list[ChatMessage],
        added: list[ChatMessage],
    ) -> None:
        removed_tokens = num_tokens_from_messages(removed)
        added_tokens = num_tokens_from_messages(added)
        self.token_estimate = self.token_estimate - removed_tokens + added_tokens

    def to_request(self) -> ChatCompletionRequest:
        """Return the API payload."""
        request = self.options.request.copy()
        request.messages = self.all_messages
        return request

    def to_atom(self) -> OutgoingMessage:
        """Create a Discord message containing the session's config."""
        with discord_open("atom.yaml") as (stream, file):
            content = yaml.safe_dump(
                self.options.dict(),
                sort_keys=False,
                default_flow_style=False,
            )
            stream.write(content.encode())
        report = (
            system_message()
            .set_color(Color2.green())
            .set_title("Chat session")
            .set_description(
                shorten(self.system_message, 4000, replace_whitespace=False)
                if self.system_message
                else "(No system message)",
            )
            .add_field("Token usage", self.usage_description)
        )
        return {"embeds": [report], "files": [file]}

    @overload
    async def fetch(self, stream: Literal[False] = False) -> ChatCompletionResponse:
        ...

    @overload
    async def fetch(self, stream: Literal[True]) -> AsyncIterator[ChatCompletionChunk]:
        ...

    async def fetch(self, stream=False):
        request = self.to_request()
        if request.limit_max_tokens(self.token_count_upper_bound):
            warnings.warn(
                f"max_tokens was reduced to {request.max_tokens}"
                " to avoid exceeding the token limit",
                stacklevel=2,
            )
        return await openai.ChatCompletion.acreate(
            **request.dict(),
            api_key=self.SECRETS.OPENAI_TOKEN.get_secret_value(),
            stream=stream,
        )  # type: ignore

    @classmethod
    async def from_thread(cls, thread: Thread):
        logger.info("Chat {0}: rebuilding history", thread.mention)

        async def get_options(message: Message):
            if not message.attachments:
                raise ValueError
            content = await message.attachments[0].read()
            try:
                return ChatSessionOptions(**yaml.safe_load(content))
            except (TypeError, ValueError, yaml.YAMLError) as e:
                raise ValueError from e

        session: cls | None = None

        async for message in thread.history(oldest_first=True):
            if session:
                async with session.editing:
                    await session.process_request(message)
                continue

            try:
                options = await get_options(message)
                logger.info("Chat {0}: found options {1}", thread.mention, options)
                assistant = message.author.mention
                session = cls(assistant=assistant, options=options)
            except ValueError:
                return None

        return session

    @classmethod
    def embed_to_plain_text(cls, role: str, author: str, embed: Embed) -> ChatMessage:
        embed = Embed2.upgrade(embed)
        match embed.type:
            case "article":
                document_type = "an article"
            case "gifv":
                document_type = "a GIF"
            case "image":
                document_type = "an image"
            case "link":
                document_type = "a link"
            case "rich":
                document_type = "a Markdown document"
            case "video":
                document_type = "a video"
        parts: list[str] = []
        parts.append(f"Discord: {author} sent {document_type}:\n")
        parts.append(str(embed))
        return ChatMessage(
            role=role,
            type_hint=ChatMessageType.PLAIN_TEXT,
            content="\n".join(parts),
        )

    @classmethod
    def embed_to_json_code_block(cls, embed: Embed) -> ChatMessage:
        raise NotImplementedError

    @classmethod
    async def attachment_to_plain_text(
        cls,
        role: str,
        author: str,
        attachment: Attachment,
    ) -> ChatMessage:
        content = f"Discord: {author} uploaded a file. "
        if attachment.filename:
            content = f"{content}Filename: {attachment.filename}. "
        if attachment.content_type:
            content = f"{content}Content type: {attachment.content_type}. "
        file_content = await attachment.read()
        try:
            text_content = file_content.decode("utf-8")
            content = f"{content}Content:\n\n{text_content}"
        except UnicodeDecodeError:
            content = f"{content}Content: (binary)."
        return ChatMessage(
            role=role,
            type_hint=ChatMessageType.PLAIN_TEXT,
            content=content,
        )

    @classmethod
    async def parse_message(
        cls,
        user: str,
        assistant: str,
        message: Message,
    ) -> list[ChatMessage]:
        if is_system_message(message):
            # Message is from us, ignore it
            return []

        if message.is_system():
            # Message is from Discord
            content = message.system_content
            for member in [message.author, *message.mentions]:
                content = content.replace(member.name, member.mention)
            return [
                ChatMessage(
                    role="system",
                    content=f"Discord: {content}",
                    message_id=message.id,
                )
            ]

        author = message.author
        messages: list[ChatMessage] = []

        role = "assistant" if author.mention == assistant else "user"

        if (
            message.type
            in (MessageType.chat_input_command, MessageType.context_menu_command)
            and message.interaction
        ):
            invoker = message.interaction.user.mention
            action = f"{message.interaction.name} command from {author.mention}"
            messages.append(ChatMessage(role=role, content=f"{invoker} used {action}"))

        if message.content:
            content = message.content
            if author.mention != user and author.mention != assistant:
                content = f"{author.mention} says: {content}"
            messages.append(ChatMessage(role=role, content=content))

        for embed in message.embeds:
            messages.append(cls.embed_to_plain_text(role, author.mention, embed))

        for attachment in message.attachments:
            item = await cls.attachment_to_plain_text(role, author.mention, attachment)
            messages.append(item)

        logger.debug("Parsed messages:")

        for result in messages:
            result.message_id = message.id
            logger.debug("{result}", result)

        return messages

    async def splice_messages(
        self,
        to_delete: int,
        to_insert: Message | None = None,
    ) -> bool:
        index = [*locate(self.messages, lambda m: m.message_id == to_delete)]
        if (
            to_insert
            and not to_insert.flags.loading
            and not is_system_message(to_insert)
        ):
            updated = await self.parse_message(
                self.options.request.user,
                self.assistant,
                to_insert,
            )
        else:
            updated = []
        if not index:
            removed = []
            self.messages.extend(updated)
        else:
            removed_slice = slice(min(index), max(index) + 1)
            removed = self.messages[removed_slice]
            self.messages[removed_slice] = updated
        self.estimate_token_usage(removed=removed, added=updated)
        return bool(updated)

    async def process_request(self, message: Message) -> bool:
        """Parse a Discord message and add it to the chain.

        Text messages from the user or the assistant will be saved as they are.
        Text messages from other users will be quoted in third-person.
        Multi-modal content (e.g. images) will be narrated in third-person from
        Discord's perspective (e.g. "Discord: <user> sent an image ...")

        :param message: a Discord Message object
        :type message: Message
        """
        return await self.splice_messages(message.id, message)

    def create_responses(self, text: str) -> list[OutgoingMessage]:
        """Convert text into a list of Discord messages.

        Long messages will be divided into chunks, splitting at new lines or
        sentences. Code blocks become individual messages. Code blocks too long
        for a Discord message become attachments.

        :param text: the text to convert
        :type text: str
        :return: a list of DiscordMessage typed dict
        :rtype: list[DiscordMessage]
        """
        parser = MarkdownIt()
        tokens = parser.parse(text)
        tree = SyntaxTreeNode(tokens)

        lines = text.splitlines()
        chunks: list[OutgoingMessage] = []

        for node in tree.children:
            line_begin, line_end = node.map
            block = "\n".join(filter(None, lines[line_begin:line_end]))
            block = dedent(block).strip()

            if node.type == "fence":
                if len(block) > MAX_MESSAGE_LENGTH:
                    with discord_open(f"code.{node.info}") as (stream, file):
                        stream.write(block.encode())
                    chunks.append({"files": [file]})
                else:
                    chunks.append({"content": block})

            else:
                for sentences in divide_text(
                    block,
                    maxlen=MAX_MESSAGE_LENGTH,
                    delimiter="\n.;?!",
                ):
                    chunks.append({"content": sentences})

        def is_rich_content(message: OutgoingMessage) -> bool:
            if message.get("embeds") or message.get("files"):
                return True
            content = message.get("content")
            return content is not None and content.startswith("```")

        results: list[OutgoingMessage] = []

        for group in split_at(chunks, is_rich_content, keep_separator=True):
            if not group:
                continue
            if is_rich_content(group[0]):
                results.extend(group)
                continue
            texts = filter(None, (chunk.get("content") for chunk in group))
            paragraphs = constrained_batches(texts, MAX_MESSAGE_LENGTH, strict=True)
            results.extend(({"content": "\n".join(p)} for p in paragraphs))

        return results

    def prepare_replies(
        self,
        response: ChatCompletionResponse,
    ) -> list[OutgoingMessage]:
        self.token_usage = response["usage"]["total_tokens"]

        if not response["choices"]:
            return []

        choice = response["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice["finish_reason"]

        results = self.create_responses(text)

        logger.info(
            "Parsed a completion response of length {length} into {num} messages",
            length=len(text),
            num=len(results),
        )

        if finish_reason:
            logger.info("Finish reason: {0}", finish_reason)
            if finish_reason != "stop":
                warnings.warn(f'Finish reason was "{finish_reason}"', stacklevel=2)

        return results

    def should_answer(self, message: Message):
        # disregard all Discord notifications
        if message.is_system():
            return

        # ignore all messages that start with a mention of another user
        # (like how tweets starting with @ are not shown to followers)
        for user in message.mentions:
            if user.mention == self.assistant:
                continue
            if message.content.startswith(user.mention):
                return False

        features = self.options.features
        # never respond to self
        result = not message.author.mention == self.assistant
        if features.response_timing == "when mentioned":
            result = result and self.assistant in [m.mention for m in message.mentions]
        if not features.respond_to_bots:
            result = result and not message.author.bot
        return result

    async def answer(
        self,
        channel: Messageable,
    ) -> bool:
        async with report_warnings(channel):
            logger.info("Sending API request")

            try:
                async with channel.typing():
                    response = await self.fetch()
            except Exception as e:
                await report_error(e, messageable=channel)
                return False

            replies = self.prepare_replies(response)
            logger.info("Resolved {0} replies", len(replies))

            for reply in replies:
                await send_message(channel, reply)

            self.warn_about_token_limit()
            return True

    async def read_chat(self, message: Message) -> bool:
        new_message = await self.process_request(message)

        if not new_message or not self.should_answer(message):
            return False

        return await self.answer(message.channel)

    def warn_about_token_limit(self):
        limit = CHAT_MODELS[self.options.request.model]["token_limit"]
        percentage = self.token_count_upper_bound / limit
        if percentage > 0.75:
            warnings.warn(
                f"Token usage is at {percentage * 100:.0f}% of the model's limit.",
                stacklevel=2,
            )

    async def write_title(self):
        ad_hoc = ChatSession(
            assistant="assistant",
            options=ChatSessionOptions(
                request=ChatCompletionRequest(
                    model="gpt-4",
                    max_tokens=64,
                    temperature=0.5,
                ),
            ),
        )
        prompt = (
            "Role: Copy editor"
            "\nTask: The following conversation has been edited into a news article."
            "\nPlease write an attractive title for it."
            "\nRequirements: Should be in the conversation's original language;"
            " Must be a single sentence or phrase"
            "\nConversation:"
        )
        messages = "\n".join(
            [f"{m.role}: {m.content}" for m in self.messages if m.role != "system"]
        )
        ad_hoc.messages.append(ChatMessage(role="user", content=prompt))
        ad_hoc.messages.append(ChatMessage(role="user", content=messages))
        ad_hoc.messages.append(ChatMessage(role="user", content="Answer:"))
        try:
            response = await ad_hoc.fetch()
        except Exception as e:
            await report_error(e)
            return
        answer = first(ad_hoc.prepare_replies(response), None)
        if answer and (title := answer.get("content")) and title:
            if unquoted := re.match(r"([\"']?)(.*)\1", title):
                return unquoted[2]
            return title
        return None
