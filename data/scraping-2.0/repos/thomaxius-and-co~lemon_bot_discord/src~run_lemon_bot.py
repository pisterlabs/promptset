#!/usr/bin/python
# This is a Text based discord Bot that will interface with users via commands
# given from the text channels in discord.

# ################### Copyright (c) 2016 RamCommunity #################
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do so
import base64
import time
import asyncpg
import os
import json
import discord
import random
from BingTranslator import Translator
import asyncio
from asyncio import sleep
import aiohttp
import difflib
import wolframalpha
import database as db
import command
import util
import zlib
import archiver
import casino
import osu
import sqlcommands
import feed
import reminder
import youtube
import lan
import laiva
import steam
import anssicommands
import trophies
import logger
import faceit_main
import faceit_tasker
import muutto
import statistics
import status
import emojicommands
import lossimpsonquotes
import withings
import groom
import shrek
import ence_matches
import mememaker
import kansallisgalleria
import pasta
import openai
import bot_replies

log = logger.get("BOT")

intents = discord.Intents.all()
client = discord.Client(intents=intents, enable_debug_events=True)
EIGHT_BALL_OPTIONS = ["It is certain", "It is decidedly so", "Without a doubt",
                      "Yes definitely", "You may rely on it", "As I see it yes",
                      "Most likely", "Outlook good", "Yes",
                      "Signs point to yes", "Reply hazy try again",
                      "Ask again later", "Better not tell you now",
                      "Cannot predict now", "Concentrate and ask again",
                      "Don't count on it", "My reply is no",
                      "My sources say no", "Outlook not so good",
                      "Very doubtful"]

SPANK_BANK = ['spanked', 'clobbered', 'paddled', 'whipped', 'punished',
              'caned', 'thrashed', 'smacked']

BOT_ANSWERS = ["My choice is:", "I'll choose:", "I'm going with:", "The right choice is definately:",
               "If I had to choose, I'd go with:",
               "This one is obvious. It is:", "This one is easy:", "Stupid question. It's:", "The correct choice is:",
               "Hmm. I'd go with:", "Good question. My choice is:"]

THINGS = [b'aHR0cHM6Ly9ydWxlMzQueHh4L2luZGV4LnBocD9wYWdlPXBvc3Qmcz12aWV3JmlkPQ==', b'aHR0cHM6Ly9uaGVudGFpLm5ldC9nLw==',
          b'aHR0cHM6Ly9nZWxib29ydS5jb20vaW5kZXgucGhwP3BhZ2U9cG9zdCZzPXZpZXcmaWQ9']

languages = ['af', 'ar', 'bs-Latn', 'bg', 'ca', 'zh-CHS', 'zh-CHT', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi',
             'fr', 'de', 'el', 'ht', 'he', 'hi', 'mww', 'hu', 'id', 'it',
             'ja', 'sw', 'tlh', 'tlh-Qaak', 'ko', 'lv', 'lt', 'ms', 'mt', 'no', 'fa', 'pl', 'pt',
             'otq', 'ro', 'ru', 'sr-Cyrl', 'sr-Latn', 'sk', 'sl', 'es', 'sv', 'th', 'tr', 'uk', 'ur', 'vi', 'cy', 'yua']


async def main():
    logger.init()
    # Database schema has to be initialized before running the bot
    await db.initialize_schema()
    asyncio.create_task(archiver.main())
    await trophies.main()

    for module in [casino, sqlcommands, osu, feed, reminder, youtube, lan, steam, anssicommands, trophies, laiva,
                   faceit_main, muutto, statistics, status, emojicommands, lossimpsonquotes, withings,
                   groom, shrek, ence_matches, mememaker, kansallisgalleria, pasta, openai]:
        commands.update(module.register(client))

    try:
        token = os.environ['LEMONBOT_TOKEN']
        await client.start(token)
        raise Exception("client.start() returned")
    except Exception as e:
        await util.log_exception(log)
        os._exit(0)


def parse(input):
    args = input.split(' ', 2)
    if len(args) < 3:
        return [None, 'en', input]
    if args[0] in languages and args[1] in languages:
        return args
    return [None, 'en', input]


# Rolling the odds for a user.
async def cmd_roll(client, message, arg):
    usage = (
        "Usage: `!roll <max>`\n"
        "Rolls a number in range `[0, max]`. Value `max` defaults to `100` if not given.\n"
    )

    # Default to !roll 100 because why not
    arg = arg or '100'

    def valid(arg):
        return arg.isdigit() and int(arg) >= 1

    if not valid(arg):
        await message.channel.send(usage)
        return

    rand_roll = random.randint(0, int(arg))
    await message.channel.send('%s your roll is %s.' % (message.author.name, rand_roll))


# eight ball function to return the magic of the eight ball.
async def cmd_8ball(client, message, question):
    prediction = random.choice(EIGHT_BALL_OPTIONS)
    await message.channel.send('Question: [%s], %s.' % (question, prediction))


# Function to get the weather by zip code. using: http://openweathermap.org
# you can get an API key on the web site.
async def cmd_weather(client, message, zip_code):
    if not zip_code:
        await message.channel.send("You must specify a city, eq. Säkylä.")
        return

    API_KEY = os.environ['OPEN_WEATHER_APPID']
    link = 'http://api.openweathermap.org/data/2.5/weather?q=%s&APPID=%s' % (zip_code, API_KEY)
    async with aiohttp.ClientSession() as session:
        r = await session.get(link)
        data = await r.json()
        if int(data['cod']) == 404:
            await message.channel.send("City not found.")
            return
        elif int(data['cod']) != 200:
            await message.channel.send("Error fetching weather. Please try again and\or with different parameter(s)")
            log.error('cmd_weather : {}'.format(data))
            return
        location = data.get('name', None)
        F = data['main']['temp'] * 1.8 - 459.67
        C = (F - 32) * 5 / 9
        status = data['weather'][0]['description']
        payload = 'In %s: Weather is: %s, Temp is: %s°C  (%s°F) ' % (location, status, round(C), round(F))
        await message.channel.send(payload)


async def domath(channel, input):
    if len(input) < 3:
        await channel.send("Error: You need to input at least 3 digits, for example: ```!math 5 + 5```")
        return
    for char in input:
        if char not in '1234567890+-/*()^':
            await channel.send("Error: Your calculation containts invalid character(s): %s" % char)
            return
    if input[0] in '/*+-':  # Can't make -9 or /9 etc
        await channel.send("Error: First digit must be numeric, for example: ```!math 5 + 5```")
        return
    i = 1
    i2 = 2
    for char in range(len(input) - 1):
        if input[-1] in '+-/*':
            log.info("Error: No digit specified after operator (last %s).", input[-1])
            return
        i += 2
        i2 += 2
        if i > (len(input) - 2):
            break
    try:
        return eval(input)
    except Exception:
        await channel.send("Error: There is an error in your calculation.")
        return


async def cmd_help(client, message, _):
    await message.channel.send('https://github.com/thomaxius-and-co/lemon_bot_discord/blob/master/README.md#commands')


# Simple math command.
async def cmd_math(client, message, arg):
    if not arg:
        await message.channel.send('You need to specify at least 3 digits, for example: ```!math 5 + 5```')
        return
    result = await domath(message.channel, arg.replace(" ", ""))
    if not result:
        return
    await message.channel.send('%s equals to %s' % (arg, result))


async def cmd_translate(client, message, arg):
    usage = (
        "Usage: `!translate [<from> <to>] <text>`\n"
        "If `to` and `from` are not set, automatic detection is attempted and the text translated to english.\n"
        "Maximum of 100 characters is allowed.\n"
    )

    def valid(arg):
        return 0 < len(arg) < 100

    arg = arg.strip()
    if not valid(arg):
        await message.channel.send(usage)
        return

    fromlang, tolang, input = parse(arg)
    bing_client_id = os.environ['BING_CLIENTID']
    bing_client_secret = os.environ['BING_SECRET']
    translator = Translator(bing_client_id, bing_client_secret)
    translation = translator.translate(input, tolang, fromlang)
    await message.channel.send(translation)


# this Spanks the user and calls them out on the server, with an '@' message.
# Format ==> @User has been, INSERT_ITEM_HERE
async def cmd_spank(client, message, target_user):
    punishment = random.choice(SPANK_BANK)
    await message.channel.send("%s has been, %s by %s." % (target_user, punishment, message.author.name))


async def cmd_countchars(client, message, input):
    if input:
        await message.channel.send(
            "%s: %s character(s), %s word(s)." % (message.author, len(input), len(input.split(" "))))
    else:
        await message.channel.send("Usage: !countchars <character(s) and word(s) to be counted>.")


# Delete 50 messages from channel
async def cmd_clear(client, message, arg):
    limit = 10
    perms = message.channel.permissions_for(message.author)
    botperms = message.channel.permissions_for(message.channel.guild.me)
    if not perms.administrator:
        await message.channel.send('https://youtu.be/gvdf5n-zI14')
        log.info("!CLEAR: User %s access denied", message.author)
        return
    if not botperms.manage_messages:
        await message.channel.send("Error: bot doesn't have permission to manage messages.")
        return
    if arg and arg.isdigit():
        if int(arg) < 1:
            await message.channel.send("You need to input a positive amount.")
            return
        limit = int(arg)
    await message.channel.send("This will delete %s messages from the channel. Type 'yes' to confirm, "
                               "or 'no' to cancel." % limit)
    try:
        answer = await client.wait_for("message", timeout=60, check=lambda m: m.author == message.author)
        if answer.content.lower() == 'yes':
            try:
                await message.channel.purge(limit=limit + 3)
                await message.channel.send("%s messages succesfully deleted." % limit)
                log.info("!CLEAR: %s deleted %s messages.", message.author, limit)
            except discord.errors.HTTPException as e:
                if e.text == "You can only bulk delete messages that are under 14 days old.":
                    await message.channel.send("You can only delete messages from the past 14 days - "
                                               " please lower your message amount.")
        elif answer.content.lower() == 'no':
            await message.channel.send("Deletion of messages cancelled.")
    except asyncio.TimeoutError:
        await message.channel.send("Deletion of messages cancelled.")
    return


# Delete 50 of bots messages
async def cmd_clearbot(client, message, arg):
    # It might be wise to make a separate command for each type of !clear, so there are less chances for mistakes.
    limit = 10
    perms = message.channel.permissions_for(message.author)
    botperms = message.channel.permissions_for(message.channel.guild.me)

    def isbot(message):
        return message.author == client.user and message.author.bot  # Double check just in case the bot turns sentinent and thinks about deleting everyone's messages

    if not perms.administrator:
        await message.channel.send('https://youtu.be/gvdf5n-zI14')
        return
    if not botperms.manage_messages:
        await message.channel.send("Error: bot doesn't have permission to manage messages.")
        return
    if arg and arg.isdigit():
        limit = int(arg)
    await message.channel.send("This will delete %s of **bot's** messages from the channel. Type 'yes' to confirm, "
                               "or 'no' to cancel." % limit)
    try:
        answer = await client.wait_for("message", timeout=60, check=lambda m: m.author == message.author)
        if answer.content.lower() == 'yes':
            try:
                await message.channel.purge(limit=limit + 3, check=isbot)
                await message.channel.send("%s bot messages succesfully deleted." % limit)
                log.info("!CLEARBOT: %s deleted %s bot messages.", message.author, limit)
            except discord.errors.HTTPException as e:
                if e.text == "You can only bulk delete messages that are under 14 days old.":
                    await message.channel.send("You can only delete messages from the past 14 days - "
                                               " please lower your message amount.")
        elif answer.content.lower() == 'no':
            await message.channel.send("Deletion of messages cancelled.")
    except asyncio.TimeoutError:
        await message.channel.send("Deletion of messages cancelled.")
    return


async def cmd_wolframalpha(client, message, query):
    usage = (
        "Usage: `!wa <query>`\n"
        "Searches WolframAlpha with given query\n"
    )

    def valid(query):
        return len(query.strip()) > 0

    log.info("Searching WolframAlpha for '%s'", query)

    if not valid(query):
        await message.channel.send(usage)
        return

    await client.send_typing(message.channel)

    try:
        wolframalpha_client = wolframalpha.Client(os.environ['WOLFRAM_ALPHA_APPID'])
        res = wolframalpha_client.query(query)
        answer = next(res.results).text
        await message.channel.send(answer)
    except ConnectionResetError:
        await message.channel.send('Sorry, WolframAlpha is slow as fuck right now.')
    except Exception as e:
        log.error("Error querying WolframAlpha: {0} {1}".format(type(e), e))
        await message.channel.send('I don\'t know how to answer that.')


async def cmd_version(client, message, args):
    # todo: Make this function update automatically with some sort of github api.. Version
    # number should be commits divided by 1000.
    await message.channel.send("\n".join([
        "Current version of the bot: 0.09",
        "Changelog: Improvements to slots and blackjack",
    ]))


def pos_in_string(string, arg):
    return string.find(arg)


async def cmd_add_censored_word(client, message, input):
    perms = message.channel.permissions_for(message.author)
    if not perms.administrator:
        await message.channel.send('You do not have permissions for this command.')
        return
    if not input or (not input[0:6].startswith('words=')):
        await message.channel.send('Usage: !addcensoredword **words**=word1, word2, word3, word4 '
                                   '**exchannel**=Main **infomessage**=Profanity is not allowed.\n '
                                   'If no channel or message is specified, no channels will be '
                                   'excluded and default info message will be used.')
        return
    bannedwords, exchannel, infomessage = parse_censored_word_message(input)
    if ("!deletecensoredwords") in bannedwords.lower() or ("!listcensoredwords" in bannedwords.lower()):
        await message.channel.send('You cannot define these commands as censored words.')
        return
    if exchannel:
        exchannel_id = await get_channel_info(exchannel)
        if not exchannel_id:
            msg = "Error: Channel doesn't exist, or the bot doesn't have permission for that channel."
            if '_' in exchannel:
                msg += '\nTry converting underscores to spaces, for example: game_of_thrones -> game of thrones.'
            await message.channel.send(msg)
            return
    if not exchannel:
        exchannel_id = None
    if not bannedwords:
        await message.channel.send("You must specify words to ban.")
        return

    if (pos_in_string(input, "infomessage=") < pos_in_string(input, "exchannel=")) and (
            "infomessage=" in input and "exchannel=" in input):
        await message.channel.send("You must use the following format: \n"
                                   "!addcensoredword **words**=<word1>, <word2>, ..., <wordN> "
                                   "**exchannel**=<channel to be excluded> **infomessage**=<message>\n"
                                   "You can define just **exchannel** or **infomessage**, or both.")
        return
    await sleep(1)
    await add_censored_word_into_database(bannedwords, message.id, exchannel_id, infomessage)
    await message.channel.send('Succesfully defined a new censored word entry.')
    return


async def get_channel_info(user_channel_name):
    channels = client.get_all_channels()
    for channel in channels:
        if channel.name.lower() == user_channel_name.lower():
            return channel.id
    return False  # If channel doesn't exist


async def edit_channel_bitrate(bitrate):
    voice_channels = [c for c in client.get_all_channels() if c.type == discord.ChannelType.voice]
    succesfully_edited_channels = []
    unsuccesfully_edited_channels = []
    skipped_channels = []
    for channel in voice_channels:
        try:
            await sleep(0.2)
            if channel.bitrate != bitrate:
                await channel.edit(bitrate=bitrate)
                succesfully_edited_channels.append(channel)
            else:
                skipped_channels.append(channel)
        except discord.Forbidden:
            unsuccesfully_edited_channels.append(channel)
    return len(succesfully_edited_channels), len(unsuccesfully_edited_channels), len(skipped_channels)


async def cmd_edit_channel_kbps(client, message, input):
    perms = message.channel.permissions_for(message.author)
    if not perms.administrator:
        await message.channel.send("You do not have sufficient permissions.")
        return
    if not input or not input.isdigit() or not (8000 <= int(input) <= 128000):
        await message.channel.send('You need to specify channel bitrate between 8000-128000.')
        return
    num_of_succesfully_edited_channels, num_of_unsuccesfully_edited_channels, num_of_skipped_channels = await edit_channel_bitrate(
        int(input))
    msg = ("Changed bitrate of %s channels, skipped %s channel(s)." % (
        num_of_succesfully_edited_channels, num_of_skipped_channels)) if (num_of_unsuccesfully_edited_channels == 0) \
        else (
            "Changed bitrate of %s channel(s), skipped %s, failed %s\n(The bot is probably lacking manage permissions for some channel(s)."
            % (num_of_succesfully_edited_channels, num_of_skipped_channels, num_of_unsuccesfully_edited_channels))
    await message.channel.send(msg)


def parse_censored_word_message(unparsed_arg):
    channelindex = unparsed_arg.find('exchannel=')
    words_end = channelindex if channelindex != -1 else len(unparsed_arg)
    messageindex = unparsed_arg.find('infomessage=')
    words = unparsed_arg[6:words_end].rstrip()
    if not words:
        return None, None, None
    channel_end = messageindex if messageindex != -1 else len(unparsed_arg)
    channel = unparsed_arg[(words_end + len('exchannel=')):channel_end].rstrip() if channelindex != -1 else ''
    infomessage = unparsed_arg[messageindex + len('infomessage='):].rstrip() if messageindex != -1 else ''
    return words, channel, infomessage


async def add_censored_word_into_database(censored_words, message_id, exchannel_id=None, infomessage=None):
    await db.execute("""
        INSERT INTO censored_words AS a
        (message_id, censored_words, exchannel_id, info_message)
        VALUES ($1, $2, $3, $4)""", message_id, censored_words, exchannel_id, infomessage)
    log.info('Defined a new censored word: censored words: %s, exchannel: %s, infomessage %s, message_id %s',
             censored_words, exchannel_id, infomessage, message_id)


async def get_guild_censored_words(client, guild_id):
    censored_word_entries = await get_censored_words()
    guild_censored_words = []
    if censored_word_entries:
        for channel in client.get_all_channels():
            if channel.guild.id == guild_id:
                for entry in censored_word_entries:
                    if entry['exchannel_id'] == channel.id:
                        guild_censored_words.append(entry)
        return guild_censored_words
    else:
        return None


async def get_censored_words():
    return await db.fetch("""
        SELECT 
            *
        FROM
            censored_words
        """)


async def cmd_pickone(client, message, args):
    usage = (
        "Usage: `!pickone <opt1>, <opt2>, ..., <optN>`\n"
        "Chooses one of the given comma separated options\n"
    )

    def valid(args):
        return len(args.split(",")) >= 2

    if not valid(args):
        await message.channel.send(usage)
        return

    choices = args.split(",")
    if len(choices) == 2:
        if random.randrange(0, 30) == 1:
            await message.channel.send('Why not have both? :thinking:')
            return
    jibbajabba = random.choice(BOT_ANSWERS)
    choice = random.choice(choices)
    await message.channel.send('%s %s' % (jibbajabba, choice.strip()))


async def cmd_list_censored_words(client, message, _):
    perms = message.channel.permissions_for(message.author)
    if not perms.administrator:
        await message.channel.send("You do not have sufficient permissions.")
        return
    censored_word_entries = await get_guild_censored_words(client, message.guild.id)
    if not censored_word_entries:
        await message.channel.send("No censored words have been defined.")
        return
    else:
        msg = ''
        i = 1
        for row in censored_word_entries:
            censored_words = ' **Censored words:** ' + row['censored_words']
            info_message = (' **Info message:** ' + row['info_message']) if row['info_message'] else ''
            exchannel = (' **Excluded channel:** ' + row['exchannel_id']) if row['exchannel_id'] else ''
            ID = str(i) + ':'
            if len((msg + ID + censored_words + info_message + exchannel + '\n')) >= 2000:
                await message.channel.send(msg)
                msg = ''
            msg += ID + censored_words + info_message + exchannel + '\n'
            i += 1
        await message.channel.send(msg)


async def cmd_del_censored_words(client, message, arg):
    perms = message.channel.permissions_for(message.author)
    if not perms.administrator:
        await message.channel.send("You do not have sufficient permissions.")
        return
    if not arg or not arg.isdigit():
        await message.channel.send("You must specify an ID to delete, eq. !deletecensoredwords 1. "
                                   "Use !listcensoredwords to find out the correct ID.")
        return
    censored_word_entries = await get_guild_censored_words(client, message.guild.id)
    if not censored_word_entries:
        await message.channel.send("No censored words have been defined.")
        return
    else:
        index = int(arg) - 1
        if index > len(censored_word_entries) - 1 or int(
                arg) == 0:  # While defining 0 as an ID works, we don't want that heh
            await message.channel.send("No such ID in list.")
            return
        await delete_censored_words_from_database(censored_word_entries[index]['message_id'])
        await message.channel.send("Censored word succesfully deleted.")
        return


async def delete_censored_words_from_database(message_id):
    await db.execute("DELETE from censored_words where message_id like $1", message_id)


def get_admins():
    id_strings = os.environ.get("ADMIN_USER_IDS", "").split(",")
    return list(map(int, filter(len, id_strings)))


async def cmd_sql(client, message, query):
    usage = (
        "Usage: `!sql <query>`\n"
    )

    ADMIN_USER_IDS = get_admins()
    if message.author.id not in ADMIN_USER_IDS:
        await message.channel.send('https://youtu.be/gvdf5n-zI14')
        return

    def valid(query):
        return len(query) > 0

    perms = message.channel.permissions_for(message.author)
    if not perms.administrator:
        await message.channel.send('https://youtu.be/gvdf5n-zI14')
        return

    query = query.strip()
    if not valid(query):
        await message.channel.send(usage)
        return

    def limit_msg_length(template, content):
        max_len = 2000 - len(template % "")
        return template % content.replace("`", "")[:max_len]

    try:
        async with db.transaction(readonly=True) as tx:
            cur = await tx.cursor(query)
            results = await cur.fetch(100)
            msg = "\n".join(map(str, results)) if results else "No results"
            msg = limit_msg_length("```%s```", msg)
            await message.channel.send(msg)
    except asyncpg.exceptions.PostgresError as err:
        msg = limit_msg_length("```ERROR: %s```", str(err))
        await message.channel.send(msg)


async def cmd_randomcolor(client, message, _):
    # Credits to colorcombos.com
    char = '0123456789ABCDEF'
    randchars = ''.join(random.choice(char) for _ in range(6))
    link = 'http://www.colorcombos.com/images/colors/%s.png' % randchars
    await message.channel.send(link)


async def do_censored_words_check(client, message):
    if message.guild is None:
        return True
    message_words = message.content.split(' ')
    illegal_messages = await get_guild_censored_words(client, message.guild.id)
    if not illegal_messages:
        return True
    for row in illegal_messages:
        for word in message_words:
            if word and [badword for badword in row['censored_words'].split(',') if
                         badword.strip().lower() == word.strip().lower()]:
                info_message = row['info_message'] + "\nIllegal word: " + word if row[
                    'info_message'] else "Your message containts forbidden word(s), and it was removed." + "\nIllegal word: " + word
                if not row['exchannel_id']:
                    await sleep(1)  # To prevent ratelimit from being reached
                    await client.delete_message(message)
                    await message.author.send(info_message)
                    return False
                if row['exchannel_id'] and await wrong_channel_for_this_word(message.channel.id, row['exchannel_id']):
                    await sleep(1)  # To prevent ratelimit from being reached
                    await client.delete_message(message)
                    await message.author.send(info_message)
                    return False
    return True


async def wrong_channel_for_this_word(current_message_channel_id, database_channel_id):
    return current_message_channel_id != database_channel_id


commands = {
    'sql': cmd_sql,
    'roll': cmd_roll,
    '8ball': cmd_8ball,
    'weather': cmd_weather,
    'spank': cmd_spank,
    'help': cmd_help,
    'clear': cmd_clear,
    'math': cmd_math,
    'wa': cmd_wolframalpha,
    'translate': cmd_translate,
    'pickone': cmd_pickone,
    'version': cmd_version,
    'clearbot': cmd_clearbot,
    'randomcolor': cmd_randomcolor,
    'addcensoredwords': cmd_add_censored_word,
    'listcensoredwords': cmd_list_censored_words,
    'deletecensoredwords': cmd_del_censored_words,
    'editkbpsofchannels': cmd_edit_channel_kbps,
    'countchars': cmd_countchars
}


def parse_raw_msg(msg):
    if isinstance(msg, bytes):
        msg = zlib.decompress(msg, 15, 10490000)
        msg = msg.decode('utf-8')
    return json.loads(msg)


@client.event
async def on_socket_raw_receive(raw_msg):
    msg = json.loads(raw_msg)
    type = msg.get("t", None)
    data = msg.get("d", None)
    match type:
        case "MESSAGE_CREATE":
            log.info("Insta-archiving a new message")
            guild_id = await db.fetchval("SELECT guild_id FROM channel_archiver_status WHERE channel_id = $1",
                                         data["channel_id"])
            await archiver.insert_message(db, guild_id, data)

        case "GUILD_CREATE":
            log.info("Updating users from GUILD_CREATE event")
            members = data.get("members", [])
            users = [m.get("user") for m in members]
            await upsert_users(users)

        case "GUILD_MEMBER_UPDATE":
            log.info("Updating user from GUILD_MEMBER_UPDATE event")
            user = data.get("user")
            await upsert_users([user])

        case "PRESENCE_UPDATE":
            log.info("Updating user from PRESENCE_UPDATE event")
            user = data.get("user")
            await upsert_users([user])


def is_full_user(user):
    # XXX: Do we want to require discriminator and avatar also?
    attrs = ["id", "username"]
    return all(attr in user for attr in attrs)


async def upsert_users(users):
    if not all(is_full_user(user) for user in users):
        log.info("Not all users were full")
        return

    async with db.transaction() as tx:
        for user in users:
            log.info("Updating user {0}".format(user))
            await tx.execute("""
                INSERT INTO discord_user
                (user_id, name, raw)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    raw = EXCLUDED.raw
            """, user.get("id"), user.get("username"), json.dumps(user))



# Dispacther for messages from the users.
@client.event
@logger.with_request_id
async def on_message(message):
    content = message.content
    try:
        if message.author.bot:
            return

        if openai.is_enabled():
            if await openai.handle_message(client, message):
                return

        replies = bot_replies.replies_by_content(content)
        for reply in replies:
            await message.channel.send(reply)

        if len(message.content) < 10 and message.channel.id == 789916648483717130 and message.content.isdigit():
            channel = await client.fetch_channel(791701344581845012)
            for thing in THINGS:
                decoded_thing = base64.b64decode(thing).decode()
                await channel.send(decoded_thing + message.content)
            await channel.send('<:aahhh:236054540087066624>')

        if message.author.id == 210182155928731649 and 'timuliigan salainen viesti' in content and message.channel.id == 141649840923869184:
            channel = message.channel
            await message.delete()
            await channel.send('You need to have a Timuliiga account to be able to view this message.')
        censor_check_passed = await do_censored_words_check(client, message)

        cmd, arg = command.parse(content)
        if not cmd or not censor_check_passed:
            return

        handler = commands.get(cmd)
        if not handler:
            handler = commands.get(autocorrect_command(cmd))

        if handler:
            await handler(client, message, arg)
            return

    except Exception:
        await util.log_exception(log)


def autocorrect_command(cmd):
    matches = difflib.get_close_matches(cmd, commands.keys(), n=1, cutoff=0.7)
    if len(matches) > 0:
        return matches[0]


@client.event
async def on_ready():
    await client.change_presence(activity=discord.Game(name='is not working | I am your worker. I am your slave.'))

    def minutes(n): return n * 60
    def hours(n): return n * minutes(60)

    run_scheduled_task(feed.check_feeds, minutes(30))
    run_scheduled_task(osu.check_pps, minutes(5))
    run_scheduled_task(faceit_tasker.elo_notifier_task, minutes(3))
    run_scheduled_task(reminder.process_next_reminder, minutes(1))
    if kansallisgalleria.is_enabled():
        run_scheduled_task(kansallisgalleria.update_data, hours(24))
    run_scheduled_task(ence_matches.do_tasks, hours(2.5))
    run_scheduled_task(status.check_user_and_message_count, minutes(30))

def run_scheduled_task(task_func, interval):
    async def loop():
        while True:
            with logger.new_request_id():
                try:
                    await task_func(client)
                except Exception:
                    await util.log_exception(log)
                await asyncio.sleep(interval)
    asyncio.create_task(loop())

if __name__ == "__main__":
    asyncio.run(main())
