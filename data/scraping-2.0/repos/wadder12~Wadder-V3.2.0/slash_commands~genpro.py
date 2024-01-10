


import nextcord
import openai


def setup(bot):
    @bot.slash_command()
    async def generate_program(interaction: nextcord.Interaction):
        """
        Generate a complex software program using the Davinci 003 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("Please provide your program requirements:", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        program_requirements = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Given the following program requirements:\n{program_requirements}\nProvide a complex software program:",
            max_tokens=1024,
            temperature=0.7,
        )
        program = response.choices[0].text.strip()
        await interaction.followup.send(embed=nextcord.Embed(title="Generated Program", description=program))  #make a code snippet 