import io

import discord
import openai

from typing import Literal

from discord import ui
from discord import app_commands
from discord.ext import commands



async def pedit(prompt, instructions, temp=0.5):
    if temp >= 1.0:
        temp: int = 1

    elif temp == 0.0 or int(temp) <= 0:
        temp: int = 0

    response = openai.Edit.create(
        model="text-davinci-edit-001",
        input=prompt,
        instruction=instructions,
        temperature=temp
    )
    return response["choices"][0]["text"]


async def imagen(prompt, size):
    # Use the OpenAI API to generate an image
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=size,
        model="image-alpha-001"
    )
    # Extract the generated image from the response
    image_url = response["data"][0]["url"]
    response = requests.get(image_url)
    image_data = response.content
    image_stream = io.BytesIO(image_data)
    return image_stream


async def completion(prompt, temp):
    if temp >= 1.0:
        temp: int = 1

    elif temp <= 0.0:
        temp: int = 0

    temp = round(temp, 1)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=temp,
        max_tokens=2048
    )

    return response["choices"][0]["text"]


class EditModal(ui.Modal, title="Message completion edit"):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    instructions:str = ui.TextInput(label='Instructions', placeholder="The prompt for the edit. Like a completion",
                                style=discord.TextStyle.paragraph, max_length=100, required=True)

    async def on_submit(self, interaction):
        try:
            await interaction.response.defer(thinking=True)
            embed = discord.Embed()
            embed.set_author(name="Text Edit")
            api_json = await pedit(self.prompt, self.instructions)
            print(api_json)
            embed.description = f"*{self.instructions}*\n\n{api_json}"

            view = ComplRegen(
                self.prompt, self.temperature, interaction.user.id)

            await interaction.edit_original_response(embed=embed, view=view)
        except Exception as e:
            print(e)


class ComplRegen(ui.View):
    def __init__(self, prompt, temperature, author):
        super().__init__()
        self.value = None
        self.prompt = prompt
        self.temperature = temperature
        self.author = author

    @ui.button(label="Regenerate", style=discord.ButtonStyle.blurple, emoji="<:regen:1060690906547765409>")
    async def complregen(self, interaction: discord.Integration, button: ui.Button):
        try:

            await interaction.response.defer(thinking=True)
            api_json = await completion(self.prompt, self.temperature)
            embed = discord.Embed()
            embed.set_author(name=f"Text Completion  ·  {self.temperature}/1")
            embed.description = f"**{self.prompt}** {api_json}"
            view = ComplRegen(self.prompt, self.temperature,
                              interaction.user.id)
            await interaction.edit_original_response(embed=embed, view=view)
        except Exception as e:
            print(e)

    @ui.button(label="Edit", style=discord.ButtonStyle.gray, emoji="<:edit:1061730605173309481>")
    async def compledit(self, interaction, button):
        try:
            await interaction.response.send_modal(EditModal(self.prompt))
        except Exception as e:
            print(e)

    async def interaction_check(self, interaction):
        if interaction.user.id != self.author:
            await interaction.response.send_message(f'This is not your menu, run </openai completion:1055108797397479475> to open your own.',
                                                    ephemeral=True)
            return False
        return True


class ImgRegen(discord.ui.View):
    def __init__(self, prompt, size, author):
        super().__init__()
        self.value = None
        self.prompt = prompt
        self.size = size
        self.author = author

    @ui.button(label="Regenerate", style=discord.ButtonStyle.blurple, emoji="<:regen:1060690906547765409>")
    async def imgregen(self, interaction, button):
        try:
            await interaction.response.defer(thinking=True)
            image_stream = await imagen(self.prompt, self.size)
            file = discord.File(image_stream, filename=f'{self.prompt}.png')

            embed = discord.Embed(
                description=f"Prompt: `{self.prompt}`, Size: `{self.size}`")
            embed.set_author(name="Text to Image")
            view = ImgRegen(self.prompt, self.size, interaction.user.id)
            await interaction.followup.send(embed=embed, file=file, view=view)
        except Exception as e:
            print(e)

    async def interaction_check(self, interaction):
        if interaction.user.id != self.author:
            await interaction.response.send_message(
                f'This is not your menu, run </openai imagen:1055108797397479475> to open your own.',
                ephemeral=True)
            return False
        return True


class OpenAI(commands.Cog):
    def __init__(self, ce):
        super().__init__()
        self.ce = ce

    openai.api_key = OPENAI

    group = app_commands.Group(
        name="openai", description="Utilize the OpenAI's api")

    @group.command(name="imagen", description="Generate images")
    @app_commands.describe(prompt="The image prompt - no default",
                           size="The image resolution - default is 512x")
    async def imagen(self, interaction, prompt: str, size: Literal['128x128', '512x512', '1024x1024'] = "512x512"):
        await interaction.response.defer(thinking=True)

        image_stream = await imagen(prompt, size)

        file = discord.File(image_stream, filename=f'{prompt}.png')

        embed = discord.Embed(
            description=f"Prompt: `{prompt}`, Size: `{size}`")
        embed.set_author(name="Text to Image")
        view = ImgRegen(prompt, size, interaction.user.id)
        try:
            await interaction.followup.send(embed=embed, file=file, view=view)
        except Exception as e:
            await interaction.followup.send(e)

    @group.command(name="completion", description="Text completion and code generation from a prompt")
    @app_commands.describe(
        prompt="The text generation prompt - no default",
        temperature="Sampling temperature. Higher values means the model will take more risks. default is 0.5")
    async def completion(self, interaction, prompt: str, temperature: float = 0.5):
        try:
            await interaction.response.defer(thinking=True)

            api_json = await completion(prompt, temperature)
            view = ComplRegen(prompt, temperature, interaction.user.id)

            embed = discord.Embed()
            embed.set_author(name=f"Text Completion  ·  {temperature}/1")
            embed.description = f"**{prompt}** {api_json}"

            await interaction.followup.send(embed=embed, view=view)
        except Exception as e:
            print(e)

    @group.command(name="edit", description="Creates a new edit for the provided input, instruction, and parameters")
    @app_commands.describe(prompt="The text the AI will work with",
                           instructions="What the AI should do with the prompt",
                           temperature="Sampling temperature. Higher values means the model will take more risks. "
                           "default is 0.5")
    async def edit(self, interaction, prompt: str, instructions: str, temperature: float = 0.5):
        await interaction.response.defer(thinking=True)

        api_json = await pedit(prompt, instructions, temperature)

        embed = discord.Embed()
        embed.set_author(name="Text Edit")
        embed.description = f"**{prompt}**\n*{instructions}*\n\n{api_json}"
        await interaction.followup.send(embed=embed)


async def setup(ce):
    await ce.add_cog(OpenAI(ce))
