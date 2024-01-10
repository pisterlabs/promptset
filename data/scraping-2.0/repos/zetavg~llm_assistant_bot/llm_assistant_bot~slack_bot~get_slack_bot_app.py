import re
import json
import logging
import asyncio
import time

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient
from langchain.memory import ConversationBufferWindowMemory
import commonmarkslack

from ..agent import Agent

from ..config import Config

TYPING_MESSAGE_TEXT = "_(thinking...)_"


def get_slack_bot_app():
    logger = logging.getLogger("slack_bot")

    # cached_agent = None

    # def get_agent() -> Agent:
    #     # nonlocal cached_agent
    #     # if cached_agent is None:
    #     #     cached_agent = Agent()
    #     # return cached_agent
    #     return Agent()

    cached_bot_info = None

    async def get_bot_info(client: AsyncWebClient):
        nonlocal cached_bot_info
        if cached_bot_info is None:
            cached_bot_info = await client.auth_test()
        return cached_bot_info

    app = AsyncApp(
        token=Config.slack.bot_user_oauth_token,
        signing_secret=Config.slack.signing_secret,
    )

    @app.event("reaction_added")
    async def reaction_added(event, client: AsyncWebClient):
        pass

    @app.event("message")
    async def message(event, client: AsyncWebClient):
        if 'bot_id' in event:
            return

        if 'subtype' in event:
            subtype = event['subtype']
            if subtype == 'message_changed':
                return

        bot_info = await get_bot_info(client)
        bot_id = bot_info["user_id"]
        bot_mention = f"<@{bot_id}>"
        bot_mention_replacement = '@bot'

        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text")
        message_ts = event.get("ts")
        # If message is not in a thread, thread_ts will be None.
        thread_ts = event.get("thread_ts", message_ts)

        is_direct_message = True

        # Check if this is a direct message by checking if the channel ID
        # starts with 'D'
        if not channel_id.startswith('D'):
            is_direct_message = False
            # In channels or group direct messages, do not reply if the bot
            # isn't mentioned.
            if f"<@{bot_id}>" not in text:
                return

        # Send a 'thinking...' message to indicate that the bot is working.
        # This message will also be used to report the execution status of
        # the bot, and will be deleted once a reply is sent.
        send_typing_message_task = asyncio.create_task(
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=message_ts,  # Should always reply in the thread.
                text=TYPING_MESSAGE_TEXT,
            )
        )

        # A function to update the 'thinking...' message to report the current
        # status.
        async def update_status_async(status):
            typing_message = await send_typing_message_task
            message_ts = typing_message['ts']
            return await client.chat_update(
                channel=channel_id,
                ts=message_ts,  # type: ignore
                text=f'_({status})_'
            )

        # A function to update the 'thinking...' message to report the current
        # status, with no async.
        def update_status(status):
            return asyncio.create_task(
                update_status_async(status)
            )

        memorized = []

        # A callback function that will be called when the agent is using a
        # tool.
        def use_tool_callback(tool_name, input):
            if tool_name == 'memorize':
                memorized.append(input)
                if len(input) > 40:
                    input = f"<{input}|{input[:40] + '...'}>"
                update_status(f'Memorizing "{input}"...')
            elif tool_name == 'check_memory':
                if len(input) > 40:
                    input = f"<{input}|{input[:40] + '...'}>"
                    update_status(f'Thinking about "{input}"...')
            elif tool_name == 'python_repl':
                update_status(f'Executing Python code...')
            elif tool_name == 'browser_google_search':
                update_status(f'Searching "{input}" on Google...')
            elif tool_name == 'browser_navigate':
                if len(input) > 40:
                    input = f"<{input}|{input[:40] + '...'}>"
                update_status(f'Browsing "{input}"...')

        agent = Agent(use_tool_callback=use_tool_callback)

        def get_info_message(time_elapsed):
            return f"\n_(Model: {agent.llm.model_name}, time elapsed: {time_elapsed:.1f}s)_"

        ai_started_at = None
        try:
            thread_replies = await client.conversations_replies(
                channel=channel_id,
                ts=thread_ts
            )

            user_info_cache = {}

            async def get_user_info(user_id):
                if user_id not in user_info_cache:
                    user_info_cache[user_id] = \
                        await client.users_info(user=user_id)
                return user_info_cache[user_id]

            async def replace_user_mentions(msg):
                # Find all mentions
                mention_ids = re.findall(r"<@([a-zA-Z0-9]+)>", msg)

                # Get user info for each mention
                user_infos = await asyncio.gather(
                    *(get_user_info(user_id) for user_id in mention_ids)
                )

                # Create a mapping from mention to user info
                mention_id_to_name = {
                    f"<@{user_id}>": f"@{info['user']['real_name']}"
                    for user_id, info in zip(mention_ids, user_infos)
                }

                # Replace each mention with its corresponding user info
                for user_id, user_name in mention_id_to_name.items():
                    msg = msg.replace(user_id, user_name)

                return msg

            history = []
            for message in thread_replies.get('messages', []):
                if message['ts'] == message_ts:
                    # Ignore the current message that triggered this event.
                    continue

                if 'bot_id' in message:
                    if (
                        message['user'] == bot_id
                        and not message['text'].startswith('_(')
                    ):
                        msg = message['text']
                        msg = await replace_user_mentions(msg)
                        history.append({
                            'from': 'bot',
                            'message': msg,
                        })
                    # Messages not from this bot are ignored.

                else:
                    user_info = await get_user_info(message['user'])
                    msg = message['text']
                    msg = message['text'].replace(
                        bot_mention, bot_mention_replacement)
                    msg = await replace_user_mentions(msg)
                    history.append({
                        'from': 'user',
                        'user_id': message['user'],
                        # 'user_id': user_info['user']['id'],
                        'user_name': user_info['user']['real_name'],
                        # 'user_display_name': user_info['user']['profile']['display_name'],
                        'message': msg,
                    })

            logger.debug(
                '---- thread history:\n%s\n---- end of thread history ----',
                json.dumps(history, indent=2, ensure_ascii=False)
            )

            memory = agent.get_new_memory()
            for h in history:
                if h['from'] == 'user':
                    memory.chat_memory.add_user_message(
                        f"@{h['user_name']}: " + h['message']
                    )
                if h['from'] == 'bot':
                    message = h['message']
                    message = re.sub(r'\n_\([^()]+\)_$', '', message)
                    message = message.strip()
                    memory.chat_memory.add_ai_message(message)

            memory.prune()
            ai_started_at = time.time()
            agent_executor = agent.get_agent_executor(
                memory=memory
            )

            user_info = await get_user_info(event['user'])
            user_name = user_info['user']['real_name']
            reply = await agent_executor.arun(
                f"@{user_name}: {text}".replace(bot_mention,
                                                bot_mention_replacement)
            )
            ai_ended_at = time.time()

            if not is_direct_message and f"@{user_name}" not in reply:
                reply = f"@{user_name} {reply}"

            reply_text = convert_markdown_to_slack(reply)

            if memorized:
                reply_text += '\n> Memorized:\n> '
                reply_text += ', '.join(memorized).replace('\n', ' ')

            reply_text += get_info_message(ai_ended_at - ai_started_at)

            return await client.chat_postMessage(
                channel=channel_id,
                thread_ts=message_ts,  # Should always reply in the thread.
                text=reply_text,
                mrkdwn=True,
                link_names=True,
            )
        except Exception as e:
            time_elapsed = 0
            if ai_started_at:
                time_elapsed = time.time() - ai_started_at

            exception = Exception(
                str(e) + f'. Event: {str(event)}'
            )
            error_message = str(e)

            if isinstance(e, asyncio.exceptions.TimeoutError):
                error_message = f"agent operation timeout (> {Config.agent.max_execution_time} seconds)"

            error_message = error_message.replace('\n', ' ')
            error_message_text = f"_⚠ An error occurred: {error_message}_"
            error_message_text += get_info_message(time_elapsed)

            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=message_ts,  # Should always reply in the thread.
                text=error_message_text,
            )

            raise exception from e
        finally:
            typing_message = await send_typing_message_task
            await client.chat_delete(
                channel=channel_id,
                ts=typing_message.get('ts', ''),
            )

    return app


def convert_markdown_to_slack(text):
    # commonmarkslack will make content in code fences disappear.
    # This is a workaround to prevent that.
    texts = text.split('```')
    # Do not convert markdown in code fences.
    texts = [
        _convert_markdown_to_slack(t) if i % 2 == 0 else t
        for i, t in enumerate(texts)]
    return '```'.join(texts)


def _convert_markdown_to_slack(text):
    parser = commonmarkslack.Parser()
    ast = parser.parse(text)
    renderer = commonmarkslack.SlackRenderer()
    slack_md = renderer.render(ast)
    return slack_md
