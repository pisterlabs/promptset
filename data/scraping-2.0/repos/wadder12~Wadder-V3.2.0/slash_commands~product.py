


import nextcord
import openai


def setup(bot):
    @bot.slash_command()
    async def generate_product_name(interaction: nextcord.Interaction):
        """
        Generate a unique product name using the Davinci 003 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("What's the product prompt?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        prompt = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Generate a unique product name based on the following prompt:\n{prompt}\nProduct name:",
            max_tokens=1024,
            temperature=0.7,
        )
        product_name = response.choices[0].text.strip()
        await interaction.followup.send(f"Here's your product name:\n```{product_name}```")