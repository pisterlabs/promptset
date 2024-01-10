import discord
import openai
import config
from discord import app_commands
from discord.ext import commands

class gpt(commands.Cog):
    def __init__(self, client: commands.Bot) -> None:
        self.client = client

    # ChatGPT Commandlines 
    @app_commands.command(name="ask", description="Ask bot the question.")
    async def ask(self, interaction: discord.Interaction, question: str):
        initial_response_embed = discord.Embed(description="Generating the answers for you, this might take a while...", color=discord.Color.blue())
        await interaction.response.send_message(embed=initial_response_embed, ephemeral=True)

        try:
            # Set the OpenAI API key using the value from config.py
            openai.api_key = config.OPENAI_API_KEY

            # Use OpenAI to generate a response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Specify the ChatGPT model to use
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ]
            )

            edited_response_embed = discord.Embed(description=response['choices'][0]['message']['content'], color=discord.Color.green())
            await interaction.edit_original_response(embed=edited_response_embed)
        except Exception as e:
            error_embed = discord.Embed(description=f"An error occurred: {e}", color=discord.Color.red())
            await interaction.edit_original_response(embed=error_embed)

async def setup(client: commands.Bot) -> None:
    await client.add_cog(gpt(client))
