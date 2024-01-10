

import nextcord
import openai

def setup(bot):
    @bot.slash_command()
    async def generate_code(interaction: nextcord.Interaction):
        """
        Generate code snippets using the Davinci 003 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("What's the problem description?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        code_description = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"{code_description}\nCode:",
            max_tokens=1024,
            temperature=0.7,
        )
        code_snippet = response.choices[0].text.strip()
        await interaction.followup.send(f"Here's some code that solves the problem:\n```{code_snippet}```")