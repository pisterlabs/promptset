import hikari
import lightbulb

from lib import openai, profanity, responses


plugin = lightbulb.Plugin("Profanity Filter")


@plugin.listener(hikari.GuildMessageCreateEvent)
async def detect_profanity_on_message(event: hikari.GuildMessageCreateEvent) -> None:
    """Checks a message for profanity. If the message contains profanity, delete it.

    Arguments:
        message: The message to check.
        message_id: The ID of the message to check.
        channel_id: The ID of the channel the message was sent in.
        guild_id: The ID of the guild the message was sent in.

    Returns:
        None.
    """
    collection = plugin.bot.d.mongo_database.settings

    response = openai.prompt(
        f"""
        Determine if this message contains profanity.

        Expected Response: YES or NO

        Message: "{event.message.content}"
        """
    )

    if response == "YES" and await profanity.is_filter_enabled(
        collection, event.guild_id
    ):
        await plugin.bot.rest.delete_message(event.channel_id, event.message.id)


@plugin.command
@lightbulb.add_checks(lightbulb.guild_only)
@lightbulb.command("toggleprofanityfilter", "Toggles the profanity filter on or off")
@lightbulb.implements(lightbulb.SlashCommand, lightbulb.PrefixCommand)
async def toggle_profanity_filter(
    context: lightbulb.SlashContext | lightbulb.PrefixContext,
) -> None:
    """The toggleprofanityfilter command. Toggles the profanity filter on or off for a guild.

    Arguments:
        context: The command context.

    Returns:
        None.
    """
    collection = plugin.bot.d.mongo_database.settings

    if await profanity.is_filter_enabled(collection, context.guild_id):
        await profanity.disable_filter(collection, context.guild_id)
        await responses.info(
            context,
            "Filter disabled",
            f"Profanity will no longer be filtered from chat.",
        )
    else:
        await profanity.enable_filter(collection, context.guild_id)
        await responses.info(
            context,
            "Filter enabled",
            f"Profanity will now be filtered from chat.",
        )


def load(bot: lightbulb.BotApp) -> None:
    """Loads the profanity filter plugin.

    Arguments:
        bot: The bot instance.

    Returns:
        None.
    """
    bot.add_plugin(plugin)
