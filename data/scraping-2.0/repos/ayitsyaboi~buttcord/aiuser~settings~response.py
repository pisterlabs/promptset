import asyncio
import json
import logging
import re
from typing import Optional

import discord
import openai
import tiktoken
from redbot.core import checks, commands
from redbot.core.utils.chat_formatting import box, pagify
from redbot.core.utils.menus import SimpleMenu, start_adding_reactions
from redbot.core.utils.predicates import ReactionPredicate

from aiuser.abc import MixinMeta, aiuser
from aiuser.common.constants import DEFAULT_REMOVELIST

logger = logging.getLogger("red.bz_cogs.aiuser")


class ResponseSettings(MixinMeta):

    @aiuser.group(name="response")
    @checks.admin_or_permissions(manage_guild=True)
    async def response(self, _):
        """ Change bot response settings

            (All subcommands are per server)
        """
        pass

    @response.command()
    @checks.is_owner()
    async def endpoint(self, ctx: commands.Context, url: Optional[str]):
        """ Sets the OpenAI endpoint to a custom one (must be OpenAI API compatible)

            Reset to official OpenAI endpoint with `[p]aiuser response endpoint clear`
        """
        if not url or url in ["clear", "reset"]:
            openai.api_base = "https://api.openai.com/v1"
            await self.config.custom_openai_endpoint.set(None)
        else:
            await self.config.custom_openai_endpoint.set(url)
            openai.api_base = url

        embed = discord.Embed(title="Bot Custom OpenAI endpoint", color=await ctx.embed_color())
        embed.add_field(
            name=":warning: Warning :warning:", value="All model/parameters selections for each server may need changing.", inline=False)

        if url:
            embed.description = f"Endpoint set to {url}."
            embed.add_field(
                name="Models", value="Third party models may have undesirable results, compared to OpenAI.", inline=False)
            embed.add_field(
                name="Note", value="This is an advanced feature. \
                    \n If you don't know what you're doing, don't use it",  inline=False)
        else:
            embed.description = "Endpoint reset back to offical OpenAI endpoint."

        await ctx.send(embed=embed)

    @response.group(name="removelist")
    @checks.admin_or_permissions(manage_guild=True)
    async def removelist(self, _):
        """ Any string in a generated response matching these regex patterns will be removed """

    @removelist.command(name="add")
    @checks.admin_or_permissions(manage_guild=True)
    async def removelist_add(self, ctx: commands.Context, *, regex_pattern: str):
        """Add a regex pattern to the removelist"""
        try:
            re.compile(regex_pattern)
        except re.error:
            return await ctx.send("Sorry, but that regex pattern seems to be invalid.")

        removelist_regexes = await self.config.guild(ctx.guild).removelist_regexes()

        if regex_pattern not in removelist_regexes:
            removelist_regexes.append(regex_pattern)
            await self.config.guild(ctx.guild).removelist_regexes.set(removelist_regexes)
            await ctx.send(f"The regex pattern `{regex_pattern}` has been added to the removelist.")
        else:
            await ctx.send(f"The regex pattern `{regex_pattern}` is already in the removelist.")

    @removelist.command(name="remove")
    @checks.admin_or_permissions(manage_guild=True)
    async def removelist_remove(self, ctx: commands.Context, *, regex_pattern: str):
        """Remove a regex pattern from the removelist"""
        removelist_regexes = await self.config.guild(ctx.guild).removelist_regexes()

        if regex_pattern in removelist_regexes:
            removelist_regexes.remove(regex_pattern)
            await self.config.guild(ctx.guild).removelist_regexes.set(removelist_regexes)
            await ctx.send(f"The regex pattern `{regex_pattern}` has been removed from the removelist.")
        else:
            await ctx.send(f"The regex pattern `{regex_pattern}` is not in the removelist.")

    @removelist.command(name="show")
    @checks.admin_or_permissions(manage_guild=True)
    async def removelist_show(self, ctx: commands.Context):
        """Show the current regex patterns in the removelist"""
        removelist_regexes = await self.config.guild(ctx.guild).removelist_regexes()
        if not removelist_regexes:
            return await ctx.send("The removelist is empty.")

        pages = []

        formatted_list = "\n".join(removelist_regexes)
        for text in pagify(formatted_list, page_length=888):
            page = discord.Embed(
                title=f"List of regexs to remove for bot messages in {ctx.guild.name}",
                description=box(text),
                color=await ctx.embed_color())
            pages.append(page)

        if len(pages) == 1:
            return await ctx.send(embed=pages[0])

        for i, page in enumerate(pages):
            page.set_footer(text=f"Page {i+1} of {len(pages)}")

        return await SimpleMenu(pages).start(ctx)

    @removelist.command(name="reset")
    @checks.admin_or_permissions(manage_guild=True)
    async def removelist_reset(self, ctx: commands.Context):
        """Reset the removelist to default """
        embed = discord.Embed(
            title="Are you sure?",
            description="This will reset this server's removelist to default.",
            color=await ctx.embed_color())
        confirm = await ctx.send(embed=embed)
        start_adding_reactions(confirm, ReactionPredicate.YES_OR_NO_EMOJIS)
        pred = ReactionPredicate.yes_or_no(confirm, ctx.author)
        try:
            await ctx.bot.wait_for("reaction_add", timeout=10.0, check=pred)
        except asyncio.TimeoutError:
            return await confirm.edit(embed=discord.Embed(title="Cancelled.", color=await ctx.embed_color()))
        if pred.result is False:
            return await confirm.edit(embed=discord.Embed(title="Cancelled.", color=await ctx.embed_color()))
        else:
            await self.config.guild(ctx.guild).removelist_regexes.set(DEFAULT_REMOVELIST)
            return await confirm.edit(embed=discord.Embed(title="Removelist reset.", color=await ctx.embed_color()))

    @response.group(name="weights", aliases=["logit_bias", "bias"])
    async def weights(self, _):
        """
            Bias the LLM for/against certain words (tokens)

            See [here](https://help.openai.com/en/articles/5247780-using-logit-bias-to-define-token-probability) for additional info.

            (All subcommands are per server)
        """
        pass

    @weights.command(name="list", aliases=["show"])
    @checks.admin_or_permissions(manage_guild=True)
    async def show_weight(self, ctx: commands.Context):
        """
            Show weights
        """
        weights = await self.config.guild(ctx.guild).weights()
        weights = {} if weights is None else json.loads(weights)
        if not weights:
            return await ctx.send(":warning: No weights set.")
        embed = discord.Embed(title="Weights Used", color=await ctx.embed_color())
        try:
            encoding = tiktoken.encoding_for_model(await self.config.guild(ctx.guild).model())
        except KeyError:
            return await ctx.send(":warning: Unsupported model for tokenization")
        weights = {encoding.decode([int(token)]): weight for token, weight in weights.items()}
        weights = {key.strip().lower(): value for key, value in weights.items()}
        for word, weight in weights.items():
            embed.add_field(name=word, value=weight, inline=False)
        await ctx.send(embed=embed)

    @weights.command(name="add")
    @checks.admin_or_permissions(manage_guild=True)
    async def set_weight(self, ctx: commands.Context, word: str, weight: int):
        """
            Sets weight for a specific word

            Will also use all possible tokens for a word when setting weight
            See [https://platform.openai.com/tokenizer](https://platform.openai.com/tokenizer) for detailed text to token conversion.

            *Arguments*
            - `word` The word to set weight for
            - `weight` The weight to set (`-100` to `100`)
        """

        custom_parameters = await self.config.guild(ctx.guild).parameters()
        if custom_parameters:
            custom_parameters = json.loads(custom_parameters)
            if custom_parameters.get("logit_bias"):
                return await ctx.send(":warning: Logit bias already set. Please remove logit bias from custom parameters first.")

        model = await self.config.guild(ctx.guild).model()
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            return await ctx.send(":warning: Unsupported model, please use custom parameters instead.")

        if weight < -100 or weight > 100:
            return await ctx.send(":warning: Weight must be between -100 and 100")

        tokens = encoding.encode(word)
        if len(tokens) > 1:
            token_str = [encoding.decode([token]) for token in tokens]
            embed = discord.Embed(
                title=f":warning: Unable to set weight for `{word}`",
                color=await ctx.embed_color()
            )
            embed.add_field(
                name=f"`{word}` is multiple tokens.\nThis means in order to modify the weight for `{word}`, all the following strings will need separate weights:",
                value=f"```{token_str}```",
                inline=False
            )
            embed.add_field(
                name="Helpful Resource",
                value=f"For better details on tokenization, go [here](https://platform.openai.com/tokenizer).",
            )
            return await ctx.send(embed=embed)
        else:
            tokens = await self.get_all_tokens(word, encoding)
            weights = await self.config.guild(ctx.guild).weights()
            weights = {} if weights is None else json.loads(weights)
            for token in tokens:
                weights[token] = weight
            await self.config.guild(ctx.guild).weights.set(json.dumps(weights))
            embed = discord.Embed(
                title=f"Weight for `{word}` set to `{weight}`",
                color=await ctx.embed_color()
            )
            return await ctx.send(embed=embed)

    @weights.command(name="remove", aliases=["delete"])
    @checks.admin_or_permissions(manage_guild=True)
    async def remove_weight(self, ctx: commands.Context, word: str):
        """
        Removes weight for a specific word

        *Arguments*
            - `word` The word to remove
        """
        try:
            encoding = tiktoken.encoding_for_model(await self.config.guild(ctx.guild).model())
        except KeyError:
            return await ctx.send(":warning: Unsupported model for tokenization")
        weights = await self.config.guild(ctx.guild).weights()
        weights = {} if weights is None else json.loads(weights)

        token = encoding.encode(word)
        if len(token) == 1:
            if str(token[0]) not in weights.keys():
                return await ctx.send(":warning: Word not found in weights.")
        else:
            return await ctx.send(":warning: Word is multiple tokens? Please check [p]aiuser response weights list")

        tokens = await self.get_all_tokens(word, encoding)
        for token in tokens:
            weights.pop(str(token))
        await self.config.guild(ctx.guild).weights.set(json.dumps(weights))
        embed = discord.Embed(
            title=f"Weight for `{word}` removed.",
            color=await ctx.embed_color()
        )
        return await ctx.send(embed=embed)

    async def get_all_tokens(self, word: str, encoding: tiktoken.Encoding):
        """
            Returns all possible tokens for a word
        """
        tokens = set()

        def append_token_if_single(token):
            encoded_token = encoding.encode(token)
            if len(encoded_token) == 1:
                tokens.add(encoded_token[0])

        append_token_if_single(word)
        append_token_if_single(word.lower())
        append_token_if_single(word.upper())
        append_token_if_single(word.capitalize())
        append_token_if_single(" " + word)
        append_token_if_single(" " + word.lower())
        append_token_if_single(" " + word.capitalize())
        append_token_if_single(" " + word.upper())
        return list(tokens)

    @response.command(name="parameters")
    @checks.is_owner()
    async def set_custom_parameters(self, ctx: commands.Context, *, json_block: str):
        """ Set custom parameters for an endpoint using a JSON code block

            To reset parameters to default, use `[p]aiuser response parameters reset`
            To show current parameters, use `[p]aiuser response parameters show`

            Example command:
            `[p]aiuser parameters ```{"frequency_penalty": 2.0, "max_tokens": 200}``` `

            See [here](https://platform.openai.com/docs/api-reference/chat/create) for possible parameters
            Some parameters are blocked.
        """
        if json_block in ['reset', 'clear']:
            await self.config.guild(ctx.guild).parameters.set(None)
            return await ctx.send("Parameters reset to default")

        embed = discord.Embed(title="Custom Parameters", color=await ctx.embed_color())
        parameters = await self.config.guild(ctx.guild).parameters()
        data = json.loads(parameters)

        if json_block not in ['show', 'list']:
            if not json_block.startswith("```"):
                return await ctx.send(":warning: Please use a code block (`` eg. ```json ``)")

            json_block = json_block.replace("```json", "").replace("```", "")

            try:
                data = json.loads(json_block)
            except json.JSONDecodeError:
                return await ctx.channel.send(":warning: Invalid JSON format!")

            blacklist = ["model", "messages", "stream"]

            invalid_keys = [key for key in data.keys() if key in blacklist]
            if invalid_keys:
                invalid_keys_str = ", ".join([f"`{key}`" for key in invalid_keys])
                return await ctx.send(f":warning: Invalid JSON! Please remove {invalid_keys_str} from your JSON.")

            if data.get("logit_bias") and await self.config(ctx.guild).weights():
                embed = discord.Embed(
                    title="Existing logit bias found!",
                    description="Wipe existing logit bias (from [p]aiuser response weights)?",
                )
                confirm = await ctx.send(embed=embed)
                start_adding_reactions(confirm, ReactionPredicate.YES_OR_NO_EMOJIS)
                pred = ReactionPredicate.yes_or_no(confirm, ctx.author)
                try:
                    await ctx.bot.wait_for("reaction_add", check=pred, timeout=30)
                except asyncio.TimeoutError:
                    return await confirm.edit(content="Canceled.")
                if pred.result is True:
                    await confirm.edit(content="Overwritten existing weights.")
                    await self.config.guild(ctx.guild).weights.set(None)
                else:
                    return await confirm.edit(content="Canceled.")

            await self.config.guild(ctx.guild).parameters.set(json.dumps(data))

        if not data:
            embed.description = "No custom parameters set."
        else:
            embed.add_field(
                name=":warning: Warning :warning:",
                value="No checks were done to see if parameters were compatible\n----------------------------------------",
                inline=False
            )
            for key, value in data.items():
                embed.add_field(name=key, value=f"```{json.dumps(value, indent=4)}```", inline=False)

        await ctx.send(embed=embed)
