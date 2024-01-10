import openai
import nextcord
from difflib import SequenceMatcher
import asyncio



def setup(bot):
    @bot.slash_command(name="quiz", description="Take a language learning quiz.")
    async def language_quiz(interaction: nextcord.Interaction, target_language: str):
        await interaction.response.send_message("Type 'quit' at any time to end the quiz.")

        while True:
            # Request a quiz question from GPT-3.5-turbo
            prompt = f"Create a language learning quiz question for {target_language}:"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates language learning quiz questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.8,
            )
            question = completion.choices[0].message["content"].strip()

            # Request the answer to the quiz question
            prompt = f"What is the answer to the following language learning quiz question: '{question}'? Provide a simple and clear answer."
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
            {"role": "system", "content": "You are a helpful assistant that provides answers to language learning quiz questions."},
            {"role": "user", "content": prompt}
        ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=1,
    )
            answer = completion.choices[0].message["content"].strip()

            # Send the question to the user
            await interaction.channel.send(question)

            # Check the user's answer and provide feedback
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            try:
                user_answer = await bot.wait_for("message", check=check, timeout=60)
                if user_answer.content.lower() == "quit":
                    await interaction.channel.send("Quiz ended. Goodbye!")
                    break

                similarity = SequenceMatcher(None, user_answer.content.lower(), answer.lower()).ratio()
                if similarity > 0.10:  # Set the similarity threshold (0.9 means 90% similarity)
                    await interaction.channel.send("Correct! Good job! 🎉")
                else:
                    await interaction.channel.send(f"Almost! The correct answer is:\n\n{answer}")
            except asyncio.TimeoutError:
                await interaction.channel.send("Time's up! Please try again.")