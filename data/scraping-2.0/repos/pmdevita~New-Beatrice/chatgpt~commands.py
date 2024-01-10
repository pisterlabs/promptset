import json
import typing
from datetime import datetime, timedelta

import alluka
import hikari.events.base_events
import openai as openai
import tanjun
import atsume

from typing import Annotated, Optional

import tiktoken
from tanjun.annotations import Member, Positional

from atsume.settings import settings

from management.schedule.ai import Schedule
from chatgpt.tools import OpenAITool

from chatgpt.prompts import prompt, should_respond_prompt

# Create your commands here.

openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_TOKEN)
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
tools: list[typing.Type[OpenAITool]] = [Schedule]
tool_names: dict[str, typing.Type[OpenAITool]] = {}
tools_compiled: list[dict] = []

for tool in tools:
    tool_names[tool.Config.name] = tool
    tools_compiled.append({
        "type": "function",
        "function": {
            "name": tool.Config.name,
            "description": tool.Config.description,
            "parameters": tool.schema()
        }
    })

print(tools, tool_names, tools_compiled)


@atsume.with_listener
async def on_message(event: hikari.events.MessageCreateEvent, client: alluka.Injected[tanjun.Client]):
    me = client.cache.get_me()
    assert me is not None
    if event.author.id == me.id:
        return
    message = event.message
    channel: hikari.GuildTextChannel = client.cache.get_guild_channel(event.channel_id)
    if not channel:
        channel = client.cache.get_thread(event.channel_id)
    assert channel is not None
    if not message.content:
        return
    # Does Beatrice want to respond?
    budget = settings.TOKEN_BUDGET
    budget -= len(enc.encode(prompt))
    message_log = []
    log = []
    async for message in channel.fetch_history(after=datetime.now() - timedelta(days=3)):
        message_log.append(message)
    for message in reversed(message_log):
        if message.author.id == me.id:
            entry = {
                "role": "assistant",
                "content": sanitize_message(message)
            }
        else:
            entry = {
                "role": "user",
                "content": f"{message.author.username}: {sanitize_message(message)}"
            }
        budget -= len(enc.encode(entry["content"]))
        log.insert(0, entry)

    log.insert(0, {"role": "system", "content": prompt})

    # Check whether to respond
    final_log = log.copy()
    final_log.append({"role": "system", "content": should_respond_prompt})
    completion = await openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=final_log, tools=tools_compiled)
    result = completion.choices[0].message.content
    should_respond = "RESPOND: YES" in result
    print(result, should_respond)
    if not should_respond:
        return

    async with channel.trigger_typing():
        completion = await openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=log, tools=tools_compiled)
        result = completion.choices[0].message.content
        tool_calls = completion.choices[0].message.tool_calls
        if tool_calls:
            tool_call(tool_calls[0])
        if result:
            print("uhoh", completion)
            if result.startswith("Beatrice:"):
                result = result[len("Beatrice:"):].lstrip()
            await channel.send(result)



async def tool_call(client: tanjun.Client, tool_call):
    tool = tool_names[tool_call.function.name]
    t = tool(json.loads(tool_call.function.arguments))
    client._injector.call_with_async_di(t.use_tool, Tool)

def sanitize_message(message: hikari.Message):
    text = message.content
    for mention in message.user_mentions.values():
        text = text.replace(f"<@{mention.id}>", mention.username)
    return text

