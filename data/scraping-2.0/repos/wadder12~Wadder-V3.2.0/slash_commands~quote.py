# working good!  

import nextcord
import openai


def setup(bot):
    @bot.slash_command(name='get_quote', description='Get you a quote from Wadder!.')
    async def generate_inspiring_quotes(interaction: nextcord.Interaction, topic: str):
        """
        Generate an inspiring quote on a specified topic using the Davinci 003 engine.
        """
        await interaction.response.defer()
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Create an inspiring quote about {topic}:",
            max_tokens=100,
            temperature=0.7,
            n=1,
            stop=None,
        )
        quote = response.choices[0].text.strip()
        embed = nextcord.Embed(title=f"Inspiring Quote on {topic.capitalize()}", description=quote, color=0x00ff00)
        await interaction.followup.send(embed=embed)  