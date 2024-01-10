import os
import openai
import asyncio
import discord
import replicate
from discord import app_commands
from discord.ext import commands
from discord.ui import Button

DISCORD_TOKEN = 'MTA3OTY1MDkyNDA5NzcyMDM0Mg.GhBdjz.b1NTpAFtydJx5MCtO8c1h1Exp7JbHXpj2rOEts'
OPEN_AI_API_KEY = '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'
REPLICATE_API_TOKEN = '57b5717e8632693a79f0747038512564a640764c'
bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())

class MyView(discord.ui.View):
    def __init__(self):
        super().__init__()

    @discord.ui.button(style=discord.ButtonStyle.green, label='A')
    async def buttonA_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Button A clicked!")

    @discord.ui.button(style=discord.ButtonStyle.red, label='B')
    async def buttonB_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Button B clicked!")

    @discord.ui.button(style=discord.ButtonStyle.red, label='C')
    async def buttonC_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Button C clicked!")

def generate_text(input_text):
    openai.api_key = OPEN_AI_API_KEY
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input_text,
        temperature=0.7,
        top_p=1,
        max_tokens=500,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
model = replicate.models.get("stability-ai/stable-diffusion")
version = model.versions.get("f178fa7a1ae43a9a9af01b833b9d2ecf97b1bcb0acfd2dc5dd04895e042863f1")

def generate_image(prompt):
    output = version.predict(prompt=prompt, negative_prompt='', width=768, height=768, prompt_strength=0.8,
                             num_outputs=1, num_inference_steps=50, guidance_scale=7.5, scheduler="DPMSolverMultistep",
                             seed=None)
    return output[0]


def image_to_image(prompt, init_image):
    output = version.predict(prompt=prompt, init_image=init_image, width=512, height=512, prompt_strength=0.8,
                             num_outputs=1, num_inference_steps=50, guidance_scale=7.5)
    return output[0]

@bot.event
async def on_ready():
    print('Bot is Up and Ready!')
    
    
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)

@bot.tree.command(name="hello")
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message(f"Hey {interaction.user.mention}!", ephemeral=True)

@bot.tree.command(name="say")
@app_commands.describe(arg="What should I say?")
async def say(interaction: discord.Interaction, arg: str):
    await interaction.response.send_message(f"{interaction.user.name} said: '{arg}'")

@bot.tree.command(name="story")
@app_commands.describe(story_prompt="Input text to get story:")
async def story(interaction: discord.Interaction, story_prompt: str):
    await interaction.response.send_message("Generating Story ........")
    await asyncio.sleep(1)
    view = MyView()
    await interaction.followup.send(generate_text(f"Write a New York Times Article describing a future where the "
                                                  f"design firm, UNFK, has made an incredible impact on the climate "
                                                  f"crisis using a combination of either product design, "
                                                  f"creative technology, or new media art. This article should be "
                                                  f"scientifically accurate and validated, and inspired by a "
                                                  f"futuristic solution that would be equitable and related to: "
                                                  f" {story_prompt}"), view=view)

@bot.tree.command(name="describe")
@app_commands.describe(desc_prompt="Input text to get description:")
async def describe(interaction: discord.Interaction, desc_prompt: str):
    await interaction.response.defer()
    await asyncio.sleep(1)
    await interaction.followup.send(generate_text(f"Describe this: {desc_prompt}"))


@bot.tree.command(name="cite")
@app_commands.describe(cite_prompt="Input text to get citations:")
async def cite(interaction: discord.Interaction, cite_prompt: str):
    await interaction.response.defer()
    await asyncio.sleep(1)
    await interaction.followup.send(generate_text(f"Provide citations of articles, specific works by designers and "
                                                  f"creatives about {cite_prompt}"))


@bot.tree.command(name="sketch")
@app_commands.describe(sketch_prompt="Input prompt for stable diffusion:")
async def sketch(interaction: discord.Interaction, sketch_prompt: str):
    await interaction.response.defer()
    await asyncio.sleep(1)
    await interaction.followup.send(generate_image(sketch_prompt))


@bot.tree.command(name="collab")
@app_commands.describe(collab_prompt="Input prompt for stable diffusion:", image="Input image for stable diffusion:")
async def collab(interaction: discord.Interaction, collab_prompt: str, image: discord.Attachment):
    await interaction.response.defer()
    await asyncio.sleep(1)
    await interaction.followup.send(image_to_image(collab_prompt, image.url))


bot.run(DISCORD_TOKEN)