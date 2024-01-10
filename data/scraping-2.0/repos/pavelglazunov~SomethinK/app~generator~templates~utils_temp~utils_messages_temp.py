UTILS_MESSAGES_IMPORT_LOG = """
from utils.event_logging import log

 """
UTILS_MESSAGES_IMPORTS_AND_BODY = """
import datetime

import disnake
import openai.error

from disnake import ApplicationCommandInteraction as AppInter
from utils.parser import load_messages, parse_config
from disnake.ext import commands


MESSAGES = load_messages()
COMMANDS = MESSAGES["commands"]
ERRORS = MESSAGES["errors"]
all_commands = list(reversed(list(parse_config("commands").keys())))
 """

UTILS_MESSAGES_GET_DESCRIPTION = """
def get_description(command):
    return COMMANDS[command]["description"]


 """
UTILS_MESSAGES_HELP_BASE = """
def get_help_description(command):
    return COMMANDS.get(command, {}).get("text_in_help", "Команда не найдена")


def get_full_description(command):
    return COMMANDS.get(command, {}).get("large_text_in_help", "Команда не найдена")


async def edit_help_page(inter: disnake.ApplicationCommandInteraction, page, message: disnake.Message = None):
    help_embed = disnake.Embed()
    help_embed.title = "Список команд"
    help_embed.description = "**Вы можете использовать /help <command> для получения более подробного описания каждой команды**"
    help_embed.set_default_colour(int("0ddddd", 16))

    pages = [all_commands[i:i + 6] for i in range(0, len(all_commands), 6)]
    if page > 1000:
        page = len(pages) - 1

    for i in pages[page]:
        help_embed.add_field(name=i, value=get_help_description(i), inline=False)

    components_list = [
        disnake.ui.Button(label="первая", style=disnake.ButtonStyle.green, custom_id=f"help_first",
                          disabled=page == 0),
        disnake.ui.Button(label="предыдущая", style=disnake.ButtonStyle.green,
                          custom_id=f"help_previous", disabled=page == 0),
        disnake.ui.Button(label=f"{page + 1}/{len(pages)}", style=disnake.ButtonStyle.secondary,
                          custom_id=f"help_page", disabled=True),
        disnake.ui.Button(label="следующая", style=disnake.ButtonStyle.green, custom_id=f"help_next",
                          disabled=page == len(pages) - 1),
        disnake.ui.Button(label="последняя", style=disnake.ButtonStyle.green,
                          custom_id=f"help_last", disabled=page == len(pages) - 1),
    ]
    if not message:
        await inter.response.send_message(
            embed=help_embed,
            components=components_list)
    else:
        await message.edit(embed=help_embed,
                           components=components_list)


def add_values(inter: AppInter, msg: str, user: disnake.Member = "", **kwargs):
    if user != "":
        msg = msg.replace("{user_name}", user.name)
        msg = msg.replace("{user_mention}", user.mention)
        msg = msg.replace("{server_name}", user.guild.name)

    if inter:
        msg = msg.replace("{author_name}", inter.user.name)
        msg = msg.replace("{author_mention}", inter.user.mention)
        msg = msg.replace("{server_name}", inter.guild.name)

    msg = msg.replace("{reason}", kwargs.get("reason", ""))
    msg = msg.replace("{result}", str(kwargs.get("result", "")))
    msg = msg.replace("{argument}", str(kwargs.get("argument", "")))
    msg = msg.replace("{reason}", str(kwargs.get("reason", "")))
    msg = msg.replace("{ping}", str(kwargs.get("ping", "")))
    msg = msg.replace("{from_language}", str(kwargs.get("from_language", "")))
    msg = msg.replace("{to_language}", str(kwargs.get("to_language", "")))
    msg = msg.replace("{text}", str(kwargs.get("text", "")))

    msg = msg.replace("{datetime}", str(datetime.datetime.now()))
    msg = msg.replace("{date}", str(datetime.datetime.now().date()))
    msg = msg.replace("{time}", str(datetime.datetime.now().time()))

    if kwargs.get("channel"):
        msg = msg.replace("{channel_name}", kwargs.get("channel").name)
        msg = msg.replace("{channel_mention}", kwargs.get("channel").mention)
    if kwargs.get("role"):
        msg = msg.replace("{role_name}", kwargs.get("role").name)
        msg = msg.replace("{role_mention}", kwargs.get("role").mention)

    return msg


 """
UTILS_MESSAGES_SEND_MESSAGE = """
async def send_long_message(inter: AppInter, command):
    command_data = COMMANDS.get(command, {})
    await inter.response.send_message(command_data["waiting_message"])


async def send_message(inter: AppInter, command, user: disnake.Member or disnake.User = "", **kwargs):
    command_data = COMMANDS.get(command, {})
    message_content = add_values(inter, command_data.get("message_text"), user, **kwargs)
    view_only_for_author = command_data.get("view_only_for_author")
    if command_data.get("send_embed"):
        embed = disnake.Embed()
        embed.title = command_data.get("embed_title")
        embed.description = message_content
        embed.set_footer(text=add_values(inter, command_data.get("embed_footer"), user, **kwargs))
        embed.color = int(command_data.get("embed_color"), 16)

        if kwargs.get("edit_original_message"):
            await inter.edit_original_message(embed=embed, content="")
            return
        await inter.response.send_message(embed=embed, ephemeral=view_only_for_author)
    else:
        if kwargs.get("edit_original_message"):
            await inter.edit_original_message(content=message_content)
            return
        await inter.response.send_message(message_content, ephemeral=view_only_for_author)
    if user:
        kwargs["user"] = user.name

     """
UTILS_MESSAGES_ADD_LOG_MESSAGE_FUNC_CALL = """
    await log(inter, f"{command}", "commands", **kwargs)


 """
UTILS_MESSAGES_SEND_EVENT_MESSAGE = """
async def send_event_message(channel, event_config: dict, member: disnake.Member, event_message_type):
    if event_config["enable_embed"]:
        embed = disnake.Embed()
        embed.title = add_values(inter="", msg=event_config["embed"]["header"], user=member)
        embed.description = add_values(inter="", msg=event_config["content"], user=member)
        embed.set_footer(text=add_values(inter="", msg=event_config["embed"]["footer_content"], user=member))
        embed.set_image(event_config["embed"]["image_url"])
        embed.color = int(event_config["embed"]["color"][1:], 16)

        await channel.send(embed=embed)
    else:
        await channel.send(add_values(inter="", msg=event_config["content"], user=member))

     """
UTILS_MESSAGES_ADD_LOG_EVENT_MESSAGE_FUNC_CALL = """
    await log(channel, event_message_type, event_message_type, **{"member": member.name})


 """
UTILS_MESSAGES_ERROR_MESSAGE_PART_1 = """
async def send_error_message(inter: AppInter, error_type, user="", **kwargs):
    error_data = ERRORS.get(error_type, {})
    message_content = add_values(inter, error_data.get("message_text"), user, **kwargs)
    error_log_text = error_data.get("logging_message")

    if user:
        kwargs["user"] = user.name
     """
UTILS_MESSAGES_ADD_LOG_ERROR_FUNC_CALL = """
    await log(inter, f"{error_type}", "ERROR", error=error_log_text, **kwargs)
"""
UTILS_MESSAGES_ERROR_MESSAGE_PART_2 = """
    
    if error_data.get("send_embed"):
        embed = disnake.Embed()
        embed.title = error_data.get("embed_title")
        embed.description = message_content
        embed.footer.text = add_values(inter, error_data.get("embed_footer"), user, **kwargs)
        embed.color = int(error_data.get("embed_color"), 16)

        if kwargs.get("edit_original_message"):
            await inter.edit_original_message(embed=embed, content="")
            return
        await inter.response.send_message(embed=embed)
    else:
        if kwargs.get("edit_original_message"):
            await inter.edit_original_message(content=message_content)
            return
        await inter.response.send_message(message_content)


 """
UTILS_MESSAGES_ERROR_DETECT = """
async def detected_error(ctx: disnake.ApplicationCommandInteraction, error_name: commands.CommandInvokeError):
    print("error handled:", error_name)
    if isinstance(error_name, commands.CommandNotFound):
        await send_error_message(ctx, "command_not_found")
    elif isinstance(error_name, commands.MissingRequiredArgument):
        await send_error_message(ctx, "missing_argument")
    elif isinstance(error_name, commands.BadArgument):
        await send_error_message(ctx, "incorrect_argument")
    elif isinstance(error_name, commands.CheckFailure):
        await send_error_message(ctx, "custom_permission_error")
    elif isinstance(error_name, disnake.Forbidden) or isinstance(error_name, disnake.InteractionTimedOut):
        await send_error_message(ctx, "time_limit_error")
    elif isinstance(error_name, openai.error.RateLimitError):
        await send_error_message(ctx, "gpt_limit_error")
    elif isinstance(error_name, commands.CommandInvokeError):
        error_name = error_name.original
        if isinstance(error_name, disnake.Forbidden):
            await send_error_message(ctx, "bot_missing_permission")
        elif isinstance(error_name, disnake.InteractionResponse):
            await send_error_message(ctx, "time_limit_error")
        else:
            await ctx.response.send_message(str(error_name))
    else:
        await send_error_message(ctx, "unknown_error")
"""
