import logging
from asyncio import Queue

import openai

from config import settings

from . import middleware
from .types import AIUsageInfo, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

incoming_queue = Queue()
outgoing_queue = Queue()


class AIBot(object):
    MIDDLEWARE_CHAIN = (
        middleware.DropIncomingIfAIDisabledMiddleware,
        middleware.StripMessageTextMiddleware,
        middleware.DropIncomingIfNotAddressedMiddleware,
        middleware.CommandParserMiddleware,
        middleware.AlternateModelSwitcherMiddleware,
        middleware.UsageCommandMiddleware,
        middleware.UsageInlineCommandMiddleware,
        middleware.HelpCommandHandlerMiddleware,
        middleware.UserDefinedPromptMiddleware,
        middleware.ContextWithPreludeMiddleware,
    )

    def __init__(self) -> None:
        openai.api_key = settings.openai.api_key

        self._middlewares = [m() for m in self.MIDDLEWARE_CHAIN]
        self._clear_context = lambda x: x

        for mw in self._middlewares:
            if isinstance(mw, middleware.ContextWithPreludeMiddleware):
                self._clear_context = mw.clear_context
                break

    async def get_completion(self, messages, model):
        result = await openai.ChatCompletion.acreate(
            model=model,
            max_tokens=settings.openai.tokens_reserved_for_response,
            messages=messages,
        )

        return result

    async def run(self):
        while True:
            message: IncomingMessage = await incoming_queue.get()

            for mw in self._middlewares:
                message = mw.incoming(message)

                if message is None or isinstance(message, OutgoingMessage):
                    break

            if message is None:
                continue

            if isinstance(message, OutgoingMessage):
                outgoing_queue.put_nowait(message)
                continue

            logger.info(f"Start AI completion for message #{message.database_id}")

            attempts = 2
            failed = False
            context_was_cleared = False

            while attempts > 0:
                try:
                    completion = await self.get_completion(message.full_with_context, message.model)
                    failed = False
                    break
                except Exception as e:
                    logger.exception(e)
                    self._clear_context(message.chat_id)
                    context_was_cleared = True
                    failed = True
                    attempts -= 1
                    continue

            if not failed:
                outgoing_text = completion.choices[0].message.content

                if context_was_cleared:
                    outgoing_text = "[token limit exceeded, context was cleared] " + outgoing_text

                outgoing_message = OutgoingMessage(
                    chat_id=message.chat_id,
                    reply_for=message.database_id,
                    text=outgoing_text,
                    model=completion.model,
                    commands=message.commands,
                    usage=AIUsageInfo(
                        prompt_tokens=completion.usage.prompt_tokens,
                        reply_tokens=completion.usage.completion_tokens,
                        total_tokens=completion.usage.total_tokens,
                    ),
                )

                for mw in reversed(self._middlewares):
                    outgoing_message = mw.outgoing(outgoing_message)

                    if outgoing_message is None:
                        break

                if outgoing_message:
                    outgoing_queue.put_nowait(outgoing_message)
