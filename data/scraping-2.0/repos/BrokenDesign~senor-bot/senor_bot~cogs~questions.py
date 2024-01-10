# type: ignore

import datetime
import re
from datetime import datetime, timedelta
from pprint import pformat

import discord
import openai
from discord import Message
from discord.ext import commands, tasks
from discord.ext.commands import Bot, Context
from icecream import ic
from loguru import logger

from senor_bot.checks import bot_manager
from senor_bot.config import settings, whitelist
from senor_bot.db import Question, write_question


class Questions(commands.Cog):
    bot: commands.Bot
    muted: bool
    open_questions: list[Question]
    whitelist: dict[int, int]

    emotes = {
        "point": "\u261D",
        "question": "\u2754",
        "check": "\u2705",
        "cross": "\u274C",
        "zipped": "\U0001F910",
        "unzipped": "\U0001F62E",
    }

    def __init__(self, bot: Bot, **kwargs):
        openai.api_key = settings.tokens.gpt
        self.bot = bot
        self.muted = False
        self.open_questions = []
        self.whitelist = whitelist
        self.question_timer_task.start()

    async def is_whitelisted(self, ctx: Context) -> bool:
        try:
            whitelisted = (
                ctx.guild.id in self.whitelist
                and ctx.channel.id == self.whitelist[ctx.guild.id]["channel"]
            )
            logger.info(f"Is whitelisted: {whitelisted}")
            return whitelisted

        except Exception as e:
            logger.error(f"Error occurred in is_whitelisted: {e}")
            return False

    async def ignore_message(self, ctx: Context) -> bool:
        try:
            result = ctx.author.id == self.bot.user.id or not await self.is_whitelisted(
                ctx
            )
            logger.info(f"Ignore Message: {result}")
            return result
        except Exception as e:
            logger.error(f"Error occurred in ignore_message: {e}")
            return False

    @commands.Cog.listener()
    async def on_message(self, ctx: Context):
        try:
            if self.muted:
                self.open_questions.clear()
                logger.debug("Bot is muted, clearing open questions.")
                return

            if await self.ignore_message(ctx):
                return

            if await self.has_open_question(ctx):
                await self.check_open_questions(ctx)

            if await self.has_question(ctx):
                if ctx.mentions[0].id == ctx.author.id:
                    logger.info("Ignoring question: self")
                    return
                if ctx.mentions[0].id == self.bot.user.id:
                    logger.info("Ignoring question: bot")
                    return
                await self.add_questions(ctx)

        except Exception as e:
            logger.error(f"Error occurred in on_message: {e}")

    @tasks.loop(minutes=1)
    async def question_timer_task(self):
        logger.info("Running question timer task...")
        await self.check_question_timers()

    async def check_question_timers(self):
        current_time = datetime.now()

        expired_questions = [
            question
            for question in self.open_questions
            if question.timer_start
            and (current_time - question.timer_start) > timedelta(minutes=5)
        ]

        if len(expired_questions) == 0:
            logger.info("No expired questions")
            return

        for question in expired_questions:
            self.open_questions.remove(question)
            write_question(question)
            logger.info(f"Removed expired question: {question.text}")

    async def has_question(self, ctx: Context) -> bool:
        try:
            result = (
                ctx.mentions is not None
                and len(ctx.mentions) == 1
                and "?" in ctx.content
            )
            logger.info(f"Message '{ctx.content}' has question: {result}")
            return result
        except Exception as e:
            logger.error(f"Error occurred in has_question: {e}")
            return False

    async def has_open_question(self, ctx: Context) -> bool:
        try:
            result = ctx.author.id in [
                question.mentions_id for question in self.open_questions
            ]
            logger.info(f"Has open question: {result}")
            return result
        except Exception as e:
            logger.error(f"Error occurred in has_open_question: {e}")
            return False

    async def strip_mentions(self, ctx: Context) -> str:
        try:
            text = ctx.content
            for user in ctx.mentions:
                text = text.replace(user.mention, "")
            logger.debug(f"Stripped mentions from text: {text}")
            return text
        except Exception as e:
            logger.error(f"Error occurred in strip_mentions: {e}")
            return text

    async def parse_questions(self, ctx: Context) -> list[str]:
        try:
            text = await self.strip_mentions(ctx)
            words = "([\w\,\-']+\s?)+"
            quote = '"[^"]*"\s?'
            terminators = "[\?\.\:\;\!]"
            pattern = f"(?P<text>({words})({quote})?({words})?)+(?P<terminator>{terminators}+|$)"
            questions = [
                match.group("text").lower().strip() + "?"
                for match in re.finditer(pattern, text)
                if match.group("text").strip() != ""
                and match.group("terminator").strip() == "?"
            ]
            logger.info(f"Parsed questions")
            logger.debug(pformat(questions))
            return questions
        except Exception as e:
            logger.error(f"Error occurred in parse_questions: {e}")
            return []

    async def add_questions(self, ctx: Context) -> None:
        try:
            questions = await self.parse_questions(ctx)
            if questions:
                await ctx.add_reaction(self.emotes["point"])
                for question in questions:
                    self.open_questions.append(Question(ctx, question))
                    logger.info(f"Added question to open questions: {question}")
            else:
                logger.info("No valid questions found in the message.")
        except Exception as e:
            logger.error(f"Error occurred in add_questions: {e}")

    async def is_answered(self, ctx: Context, question: Question) -> bool:
        try:
            prompt = f"Can '{ctx.content}' be considered an answer to the question '{question.text}'"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens=1,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content.strip().lower() == "yes"
            logger.info(f"Answer for question '{question.text}': {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error occurred in is_answered: {e}")
            raise e

    async def replies_to_open_question(self, ctx: Context) -> bool:
        try:
            if ctx.reference is None:
                result = False
            else:
                result = ctx.reference.message_id in [
                    question.message_id
                    for question in self.open_questions
                    if question.mentions_id == ctx.author.id
                ]
            logger.info(f"Replies to open question: {result}")
            return result
        except Exception as e:
            logger.error(f"Error occurred in replies_to_open_question: {e}")
            return False

    async def increment_user_questions(self, ctx: Context) -> None:
        user_questions = [
            question
            for question in self.open_questions
            if question.mentions_id == ctx.author.id
        ]
        for question in user_questions:
            question.replies += 1

    async def check_open_questions(self, ctx: Context) -> None:
        try:
            assert await self.has_open_question(ctx)
            if not await self.replies_to_open_question(ctx):
                await ctx.add_reaction(self.emotes["question"])
                await self.increment_user_questions(ctx)
                logger.info(f"Added question reaction")
            else:
                for question in self.open_questions:
                    if (
                        ctx.author.id != question.mentions_id
                        or ctx.reference.message_id != question.message_id
                    ):
                        logger.info(f"Question {question.text} is mentioned: False")
                        continue
                    try:
                        has_answer = await self.is_answered(ctx, question)
                        if not has_answer:
                            await ctx.add_reaction(self.emotes["cross"])
                            question.replies += 1
                            logger.info(
                                f"Added cross reaction for question ID: {question.message_id}"
                            )
                        else:
                            await ctx.add_reaction(self.emotes["check"])
                            question.replies += 1
                            question.answer = ctx.content
                            question.has_answer = True
                            await write_question(question)
                            self.open_questions.remove(question)
                            logger.info(
                                f"Question ID {question.message_id} successfully closed."
                            )

                    except Exception as e:
                        has_answer = False
                        await ctx.send("error: openai error, clearing open questions")
                        self.open_questions.clear()
                        logger.error(f"Error occurred in check_open_questions: {e}")
                        return

        except Exception as e:
            logger.error(f"Error occurred in check_open_questions: {e}")

    @commands.check(bot_manager)
    @commands.slash_command(
        name="mute", description="toggles checking of questions on/off"
    )
    async def mute(self, ctx: Context):
        self.muted = not self.muted
        if self.muted:
            await ctx.respond(self.emotes["zipped"])
            logger.info("Questions muted")
        else:
            await ctx.respond(self.emotes["unzipped"])
            logger.info("Questions unmuted")

    @commands.slash_command(name="list", description="lists open questions")
    async def send_open_questions(self, ctx: Context):
        embed = discord.Embed()
        embed.title = "Open Questions"
        embed.color = 10038562
        if len(self.open_questions) == 0:
            embed.description = "There are currently no open questions"
            await ctx.respond(embed=embed)
            return

        if len(self.open_questions) > 25:
            embed.description = (
                "NOTE: there are currently > 25 open questions and output has been truncated"
                "The following questions are currently waiting for answers\n_ _"
            )
            await ctx.respond(embed=embed)
        else:
            for i, question in enumerate(self.open_questions):
                author = await self.bot.fetch_user(question.author_id)
                assert author is not None
                embed.add_field(
                    name=f"Question #{i+1}",
                    value=f"<@!{question.mentions_id}> {question.text}",
                    inline=False,
                )
            await ctx.respond(embed=embed)

    @commands.check(bot_manager)
    @commands.slash_command(
        name="remove", description="removes question from list of open questions by #"
    )
    async def remove_open_question(self, ctx: Context, number: int):
        if len(self.open_questions) == 0:
            await ctx.respond("Error: No open questions")
            logger.warning("No open questions found while trying to remove a question")

        elif number not in range(1, len(self.open_questions) + 1):
            await ctx.respond(
                f"Invalid index: expected value in 1..{len(self.open_questions)}."
            )
            logger.warning(
                f"Invalid index {number} provided while trying to remove a question"
            )

        else:
            question = self.open_questions.pop(number - 1)
            embed = discord.Embed()
            embed.title = "Removed question"
            embed.color = discord.Color.dark_red()
            embed.add_field(
                name="Asker", value=f"<@!{question.author_id}>", inline=False
            )
            embed.add_field(
                name="Mentions", value=f"<@!{question.mentions_id}>", inline=False
            )
            embed.add_field(name="Question", value=question.text, inline=False)
            await ctx.respond(embed=embed)
            logger.info(f"Question {number} removed successfully")

    @commands.check(bot_manager)
    @commands.slash_command(
        name="close", description="closes an open question as answered"
    )
    async def close_open_question(self, ctx: Context, n: int, answer: str):
        if ctx.author.id != settings.bot.owner.id:
            await ctx.respond("Insufficient permission: Owner required")

        elif len(self.open_questions) == 0:
            await ctx.respond("Error: No open questions")

        elif number not in range(1, len(self.open_questions) + 1):
            await ctx.respond(
                f"Invalid index: expected value in 1..{len(self.open_questions)}."
            )

        elif answer is None or answer.strip() == "":
            await ctx.respond(f"Error: must supply answer text")

        else:
            question = self.open_questions.pop(number - 1)
            question.has_answer = True
            question.answer = answer
            await write_question(question)
            await ctx.respond(f"Closed question:\n```{pformat(question.to_dict())}```")

    @commands.check(bot_manager)
    @commands.slash_command(name="clear", description="clears *all* open questions")
    async def clear_open_questions(self, ctx: Context):
        if ctx.author.id != settings.bot.owner.id:
            await ctx.respond("Insufficient permission: Owner required")
        self.open_questions.clear()
        await self.send_open_questions(ctx)


def setup(bot: commands.Bot):
    bot.add_cog(Questions(bot))
