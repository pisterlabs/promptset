import openai
import nextcord
import asyncio



def setup(bot):
    @bot.slash_command(name="translate_practice", description="Practice translating sentences.")
    async def translation_practice(interaction: nextcord.Interaction, target_language: str):
        await interaction.response.send_message("Type 'quit' at any time to end the translation practice.")

        while True:
            # Request a sentence to translate from GPT-3.5-turbo
            prompt = f"Generate a simple sentence in English for translation practice:"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.8,
            )
            sentence = completion.choices[0].message["content"].strip()

            # Request translation of the sentence
            prompt = f"Translate the following sentence from English to {target_language}: '{sentence}'"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.8,
            )
            response = completion.choices[0].message["content"].strip()

            # Send the sentence to the user and ask for their translation
            await interaction.channel.send(f"Translate the following sentence to {target_language}:\n\n{sentence}")

            # Check the user's translation and provide feedback
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            try:
                user_translation = await bot.wait_for("message", check=check, timeout=60)
                if user_translation.content.lower() == "quit":
                    await interaction.channel.send("Translation practice ended. Goodbye!")
                    break

                if user_translation.content.lower() == response.lower():
                    await interaction.channel.send("Correct! Good job! 🎉")
                else:
                    await interaction.channel.send(f"Almost! The correct translation is:\n\n{response}")
            except asyncio.TimeoutError:
                await interaction.channel.send("Time's up! Please try again.")