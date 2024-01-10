import discord
from discord.ext import commands
from discord import app_commands
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.environ["OPENAI_API_KEY"],
)
class open_ai_cog(commands.Cog):
  def __init__(self, client: commands.Bot):
    self.client = client

  def chat_gpt_api(self, prompt):
    model_engine = "gpt-4-turbo" # or any other GPT model

    completions = client.chat.completions.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=200
    )
    return completions.choices[0].text.strip()

  def image_gpt_api(self, prompt):
    response = client.images.generate(
    prompt=prompt,
    n=1,
    size="256x256"
    )
    url = response['data'][0]['url']
    return str(url)
    

  @app_commands.command(name = "image")
  async def image(self, interaction: discord.Interaction, text: str):
    await interaction.response.defer(thinking = True)
    embed = discord.Embed(title="*" + str(interaction.user.name) + ": " + text + "*")
    embed.set_image(url=self.image_gpt_api(text))

    await interaction.followup.send(embed=embed)
    
  @app_commands.command(name = "chat")
  async def chat(self, interaction: discord.Interaction, text: str):
    await interaction.response.defer(thinking = True)
    await interaction.followup.send("*" + str(interaction.user.name) + ": " + text + "*\n\n" + self.chat_gpt_api(text))