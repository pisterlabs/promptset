

import nextcord
import openai


def setup(bot):
    @bot.slash_command()
    async def generate_story(interaction: nextcord.Interaction):
        """
        Generate a short story using the Davinci 003 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("What's the story prompt?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        prompt = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Write a short story based on the following prompt:\n{prompt}\nStory:",
            max_tokens=1024,
            temperature=0.7,
        )
        story = response.choices[0].text.strip()
        await interaction.followup.send(f"Here's your story:\n```{story}```")