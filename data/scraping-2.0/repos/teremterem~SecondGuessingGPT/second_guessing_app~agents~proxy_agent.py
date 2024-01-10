"""
This module contains a proxy agent that takes response messages from other agents and breaks them down into paragraphs,
sending each paragraph as a separate Telegram message. The same kind of splitting is also reflected in AgentForum
message tree.
"""
import asyncio

from agentforum.forum import InteractionContext, Agent

from second_guessing_app.agents.forum import tg_app, keep_typing


async def send_forum_msg_to_telegram(
    wrapped_agent: Agent,
    ctx: InteractionContext,
    tg_chat_id: int = None,
    reply_to_tg_msg_id: int | None = None,
    **kwargs,
) -> None:
    # pylint: disable=too-many-locals
    """
    Send a message from AgentForum to Telegram. Break it up into multiple Telegram messages based on the presence of
    double newlines.
    """

    async def send_tg_message(content: str) -> None:
        """
        Send a Telegram message. If `reply_to_tg_msg_id` is not None, then reply to the message with that ID and
        then set `reply_to_tg_msg_id` to None to make sure that only the first message in the series of responses
        is a reply.
        """
        nonlocal reply_to_tg_msg_id
        if reply_to_tg_msg_id is None:
            kwargs = {}
        else:
            kwargs = {"reply_to_message_id": reply_to_tg_msg_id}
            reply_to_tg_msg_id = None
        tg_message = await tg_app.bot.send_message(chat_id=tg_chat_id, text=content, **kwargs)
        ctx.respond(
            content,
            tg_message_id=tg_message.message_id,
            tg_chat_id=tg_chat_id,
            openai_role="assistant",
        )

    async for msg_promise in wrapped_agent.quick_call(ctx.request_messages, **kwargs):
        tokens_so_far: list[str] = []

        typing_task = asyncio.create_task(keep_typing(tg_chat_id))

        async for token in msg_promise:
            tokens_so_far.append(token.text)
            content_so_far = "".join(tokens_so_far)

            if content_so_far.count("```") % 2 == 1:
                # we are in the middle of a code block, let's not break it
                continue

            broken_up_content = content_so_far.rsplit("\n\n", 1)
            if len(broken_up_content) != 2:
                continue

            typing_task.cancel()

            content_left, content_right = broken_up_content

            tokens_so_far = [content_right] if content_right else []
            if content_left.strip():
                await send_tg_message(content_left)

            typing_task = asyncio.create_task(keep_typing(tg_chat_id))

        typing_task.cancel()

        remaining_content = "".join(tokens_so_far)
        if remaining_content.strip():
            await send_tg_message(remaining_content)

        # # TODO Oleksandr: what happens if I never materialize the original (full) message from openai ?
        # await msg_promise.amaterialize()
