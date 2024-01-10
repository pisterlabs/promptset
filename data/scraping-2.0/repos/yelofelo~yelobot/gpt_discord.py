import emoji
import re
import asyncio
import discord
import time

from openai_interface import OpenAIInterface


# TODO: refactor, put this elsewhere
SYSTEM_MESSAGE = "YeloBot is not an assistant, but a chatbot who tries to fit in with the other members of the chat."


async def respond_to(bot, message_or_channel, openai_interface: OpenAIInterface) -> None:
    if isinstance(message_or_channel, discord.TextChannel):
        channel = message_or_channel
    else:
        channel = message_or_channel.channel

    history = list(reversed([msg async for msg in channel.history(limit=9)]))

    response = await generate_gpt_3(bot, history, openai_interface)

    # first_line = True

    for i, line in enumerate(response.splitlines()):
        if i == 5:
            await channel.send('[I was going to keep talking, but YeloFelo is trying to be responsible so he limited me :/]')
            return
        # if first_line: # and not isinstance(message_or_channel, discord.TextChannel):
        #     await channel.send(line)
        # else:
        async with channel.typing():
            await asyncio.sleep(len(line) / 20)
        await channel.send(line)
        # first_line = False


async def generate_gpt_3(bot, messages, openai_interface: OpenAIInterface) -> str:
    global GPT_SESS

    prefix = ''

    last_author = None

    mention = f'<@!{bot.user.id}>'
    mobilemention = f'<@{bot.user.id}>'

    for message in messages:
        if 'http:' in message.content or 'https:' in message.content or message.attachments or not message.content:
            continue

        if last_author != message.author.id:
            prefix += '\n' + message.author.name + ':\n'
            last_author = message.author.id

        prefix += emoji.demojize(message.content.replace(mention, 'yelobot').replace(mobilemention, 'yelobot').replace('@', '').replace('\n\n', '\n'), language='alias') + '\n'

    prefix = reverse_replace_emote(prefix.lstrip('\n')) + f'\nYeloBot:\n'

    compiled = re.compile(r'(\<@!(?P<id1>\d+)\>)|(\<@(?P<id2>\d+)\>)')

    mo = re.search(compiled, prefix)
    while mo:
        to_rep = mo.group(0)
        if mo.group('id1'):
            uid = mo.group('id1')
        else:
            uid = mo.group('id2')
        user = await bot.fetch_user(uid)
        prefix = prefix.replace(to_rep, user.name)
        mo = re.search(compiled, prefix)

    # print(prefix)

    output = openai_interface.generate(prefix, SYSTEM_MESSAGE)

    return emoji.demojize(replace_with_emote(bot, output))


async def send_if_idle(bot, time_to_wait, channel, status, openai_interface, lock):
    start_time = time.time()

    while time.time() - start_time < time_to_wait:
        if not status['idle']:
            return False
        await asyncio.sleep(10)

    async with lock:
        if not status['idle']:
            return False
        await respond_to(bot, channel, openai_interface)
    return True


def replace_with_emote(bot, text) -> str:
    def repl(emotematch):
        e = discord.utils.get(bot.emojis, name=emotematch.group(1))
        if e:
            return '<a' + emotematch.group(0) + str(e.id) + '>' if e.animated else  '<' + emotematch.group(0) + str(e.id) + '>'
        else: 
            return emotematch.group(0)

    return re.sub(r':(.+):', repl, text)


def reverse_replace_emote(text) -> str:
    return re.sub(r'\<(:.+:)\d+\>', lambda match: match.group(1), text)
