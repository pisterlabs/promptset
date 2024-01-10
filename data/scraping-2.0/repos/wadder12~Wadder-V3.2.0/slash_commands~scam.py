

import asyncio
import nextcord

import openai


def setup(bot):
    @bot.slash_command(name="scramble", description="Play a word scramble game in the target language.")
    async def word_scramble(interaction: nextcord.Interaction, target_language: str):
        await interaction.response.send_message("Type 'quit' at any time to end the game.")

        while True:
            # Request a scrambled word from GPT-3.5-turbo
            prompt = f"Create a scrambled word for a word scramble game in {target_language}:"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates word scramble challenges."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.8,
            )
            scrambled_word = completion.choices[0].message["content"].strip()

            # Request the unscrambled word
            prompt = f"What is the unscrambled version of the following scrambled word: '{scrambled_word}'?"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that unscrambles words."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=1,
            )
            unscrambled_word = completion.choices[0].message["content"].strip()

            # Send the scrambled word to the user
            await interaction.channel.send(f"Unscramble this word: {scrambled_word}")

            # Check the user's answer and provide feedback
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            try:
                user_answer = await bot.wait_for("message", check=check, timeout=60)
                if user_answer.content.lower() == "quit":
                    await interaction.channel.send("Game ended. Goodbye!")
                    break

                if user_answer.content.lower() == unscrambled_word.lower():
                    await interaction.channel.send("Correct! Great job! 🎉")
                else:
                    await interaction.channel.send(f"Oops! The correct answer is:\n\n{unscrambled_word}")
            except asyncio.TimeoutError:
                await interaction.channel.send("Time's up! Please try again.")