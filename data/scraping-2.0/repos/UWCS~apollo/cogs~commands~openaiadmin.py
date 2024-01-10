from discord import Interaction, User
from discord.ext import commands
from discord.ext.commands import Bot, Context, check
from psycopg import OperationalError
from sqlalchemy.exc import SQLAlchemyError

from models import db_session
from models.openai import OpenAIBans
from models.user import User as db_user
from utils import get_database_user, get_database_user_from_id, is_compsoc_exec_in_guild

LONG_HELP_TEXT = """
Exec-only command to stop or reallow a user from usinf OpenAI functionality in apollo (Dalle and ChatGPT)
"""


class OpenAIAdmin(commands.Cog):
    def __init__(self, bot: Bot):
        self.bot = bot

    @commands.hybrid_group(help=LONG_HELP_TEXT, brief="ban/unban user from OpenAI")
    @check(is_compsoc_exec_in_guild)
    async def openaiadmin(self, ctx: Context):
        if not ctx.invoked_subcommand:
            await ctx.send("Subcommand not found")

    @openaiadmin.command(help=LONG_HELP_TEXT, brief="ban/unban user from OpenAI")
    @check(is_compsoc_exec_in_guild)
    async def ban(self, ctx: Context, user: User, ban: bool):
        """bans or unbans a user from OpenAI commands"""

        if user == ctx.author:
            return await ctx.reply("You can't ban yourself")

        db_user = get_database_user(user)

        if not db_user:
            return await ctx.reply("User not found please try again")

        is_banned = (  # a user is banned if in the db
            db_session.query(OpenAIBans)
            .filter(OpenAIBans.user_id == db_user.id)
            .first()
            is not None
        )
        if ban:
            if is_banned:
                return await ctx.reply("User already banned")

            banned_user = OpenAIBans(user_id=db_user.id)
            db_session.add(banned_user)
        else:
            if not is_banned:
                return await ctx.reply("User not banned")

            db_session.query(OpenAIBans).filter(
                OpenAIBans.user_id == db_user.id
            ).delete()

        db_session.commit()
        await ctx.reply(
            f"User {user} has been {'banned' if ban else 'unbanned'} from OpenAI commands"
        )

    @openaiadmin.command(help=LONG_HELP_TEXT, brief="list banned users")
    @check(is_compsoc_exec_in_guild)
    async def list(self, ctx: Context):
        """lists all users banned from OpenAI commands"""
        banned_users = db_session.query(OpenAIBans).all()  # get all users in db

        if not banned_users:
            return await ctx.reply("No users banned")

        banned_users_str = "\n".join(  # make list of names
            [
                db_session.query(db_user)
                .filter(db_user.id == user.user_id)
                .first()
                .username
                for user in banned_users
            ]
        )

        await ctx.reply(f"Users banned from OpenAI:\n{banned_users_str}")

    @openaiadmin.command(help=LONG_HELP_TEXT, brief="is user banned")
    @check(is_compsoc_exec_in_guild)
    async def is_banned(self, ctx: Context, user: User):
        """checks if user is banned from OpenAI commands"""
        is_banned = is_user_banned_openai(user.id)
        await ctx.reply(
            f"User {user} **is {'**banned' if is_banned else 'not** banned'} from using OpenAI commands"
        )


@staticmethod
async def is_author_banned_openai(ctx: Context | Interaction):
    """returns true if author is banned from OpenAI commands"""
    id = ctx.author.id if isinstance(ctx, Context) else ctx.user.id
    banned = is_user_banned_openai(id)
    if banned:
        await openai_ban_error(ctx, id)
    return not banned


@staticmethod
def is_user_banned_openai(id: int):
    """returns true if user is banned from OpenAI commands"""
    try:
        db_user = get_database_user_from_id(id)
        if not db_user:
            return False
        return (
            db_session.query(OpenAIBans)
            .filter(OpenAIBans.user_id == db_user.id)
            .first()
            is not None
        )
    except (SQLAlchemyError, OperationalError):
        return False


@staticmethod
async def openai_ban_error(ctx: Context | Interaction, id: int):
    """error for OpenAI commands"""

    message = (
        "no you horny mf <:mega_flushed:1018541064325451887>"
        if id == 274261420932202498
        else "You are banned from using OpenAI commands, please contact an exec if you think this is a mistake"
    )

    if isinstance(ctx, Context):
        await ctx.reply(message)
    else:
        await ctx.response.send_message(message)


async def setup(bot: Bot):
    await bot.add_cog(OpenAIAdmin(bot))
