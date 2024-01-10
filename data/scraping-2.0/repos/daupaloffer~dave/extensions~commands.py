import discord
import openai
import asyncio
from discord import app_commands
from discord.ext import commands


class BasicCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    async def get_gpt_completion(self, prompt, streamer: bool):
        self.bot.gpt_timeout = True
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=1.0,
                max_tokens=256,
                stream=streamer,
                messages=prompt
            )

        except openai.error.RateLimitError:
            print("Ratelimited by OpenAI!")
            return

        return completion

    async def send_gpt_stream(self, completion, interaction = None, message = None):
        collected_messages = []
        message_string = ""
        iteration = 0
        first_message = True
        new_message = None

        for chunk in completion:
            chunk_message = chunk["choices"][0]["delta"]
            finish_reason = chunk["choices"][0]["finish_reason"]

            chunk_message_plain = chunk_message.to_dict().get("content")
            if finish_reason == "stop":
                iteration = 5

            if chunk_message_plain is not None:
                if first_message:
                    if message is not None:
                        new_message = await message.reply(chunk_message_plain)
                    else:
                        await interaction.response.send_message(chunk_message_plain)

                    first_message = False

                collected_messages.append(chunk_message_plain)
                message_string = "".join(collected_messages)
                iteration += 1

            if iteration == 5:
                if message is not None:
                    await new_message.edit(content=message_string)
                else:
                    await interaction.edit_original_response(content=message_string)
                await asyncio.sleep(1)
                iteration = 0

        await asyncio.sleep(2)
        self.bot.gpt_timeout = False

    @app_commands.command(name="test-command", description="This is a test command")
    async def test_command(self, interaction: discord.Interaction):
        await interaction.response.send_message("idiot")

    @app_commands.command(name="owo", description="owo-ify your message")
    @app_commands.describe(
        message="Provide a message to be owo-ified instead of using previous message",
        anonymous="Make it look like it was Dave Prime who sent the message",
    )
    async def owoifier(
        self,
        interaction: discord.Interaction,
        message: str = "",
        anonymous: bool = False,
    ):
        if message:
            owo_payload = message
        else:
            message_history = interaction.channel.history(limit=1)
            owo_payload = [message async for message in message_history][0]

            if not owo_payload.content:  # Maybe add owo_payload.author.bot too
                await interaction.response.send_message("try again, idiot")
                return
            else:
                owo_payload = owo_payload.content

        chars_to_replace = "lrLR"
        for i in chars_to_replace:
            if i.islower():
                owo_payload = owo_payload.replace(i, "w")
            else:
                owo_payload = owo_payload.replace(i, "W")

        if anonymous:
            await interaction.channel.send(owo_payload)
            await interaction.response.send_message("Message sent!", ephemeral=True)
        else:
            await interaction.response.send_message(owo_payload)

    @app_commands.command(name="backstory", description="Create your very own character profile")
    async def backstory(
            self,
            interaction: discord.Interaction,
            member: discord.Member
    ):
        if self.bot.gpt_timeout:
            await interaction.response.send_message("try again in a couple seconds", ephemeral=True)
            return

        prompt = [
            {"role": "system", "content": "You generate a short creative character backstory based on the name of the user, using an appropriate theme."},
            {"role": "system", "content": "The backstory includes a viscerally horrific, graphic, dark twist."},
            {"role": "system", "content": "The backstory is shorter than 200 words."},
            {"role": "system", "content": "The generation only contains the backstory, no prefix."},
            {"role": "system", "content": f"Name: {member.display_name}"}
        ]

        completion = await self.get_gpt_completion(prompt, streamer=True)
        await self.send_gpt_stream(completion, interaction)

    @app_commands.command(name="berate", description="Berate a fool")
    async def berate(
            self,
            interaction: discord.Interaction,
            member: discord.Member = None
    ):
        if self.bot.gpt_timeout:
            await interaction.response.send_message("try again in a couple seconds", ephemeral=True)
            return

        message_history = interaction.channel.history(limit=10)
        messages = [message async for message in message_history]

        target_message = None
        for message in messages:
            if member is not None:
                if message.author == member:
                    target_message = message
                    break
            else:
                if message.author != interaction.user:
                    target_message = message
                    break

        if not target_message:
            await interaction.response.send_message("try again but with someone who sent a message more recently", ephemeral=True)
            return

        if target_message.attachments:
            await interaction.response.send_message("try again with a normal message", ephemeral=True)
            return

        prompt = [
            {"role": "system", "content": "You are a character simulation that simulates a specified character."},
            {"role": "system", "content": "This is an online chat between you and another user."},
            {"role": "system", "content": "Act as if you are a Monty Python character."},
            {"role": "system", "content": "Reply with a sarcastic, joking insult based on the other person's message."},
            {"role": "user", "content": f"{target_message.author.display_name}: {target_message.content}"}
        ]

        await interaction.response.defer(thinking=True)
        completion = await self.get_gpt_completion(prompt, streamer=True)
        await self.send_gpt_stream(completion, message=target_message)
        await interaction.delete_original_response()

    @app_commands.command(name="show-notes", description="Show developer notes")
    async def show_notes(self, interaction: discord.Interaction):
        file = open("notes.txt", "r")
        content = "".join(file.readlines())
        await interaction.response.send_message(content)


async def setup(bot):
    print("Loading commands extension..")
    await bot.add_cog(BasicCommands(bot))
