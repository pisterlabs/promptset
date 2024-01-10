import openai
import nextcord
import asyncio


def setup(bot):
    @bot.slash_command(name="mcquiz", description="Take a multiple-choice language learning quiz.")
    async def multiple_choice_quiz(interaction: nextcord.Interaction, target_language: str):
        await interaction.response.send_message("Type 'quit' at any time to end the quiz.")

        while True:
            # Request a multiple-choice quiz question from GPT-3.5-turbo
            prompt = f"Create a multiple-choice language learning quiz question for {target_language}:"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates multiple-choice language learning quiz questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                n=1,
                stop=None,
                temperature=0.8,
            )
            question = completion.choices[0].message["content"].strip()

            # Request the answer to the quiz question
            prompt = f"What is the correct answer to the following multiple-choice language learning quiz question: '{question}'? Provide the letter of the correct answer."
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides answers to multiple-choice language learning quiz questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                n=1,
                stop=None,
                temperature=1,
            )
            answer = completion.choices[0].message["content"].strip().lower()

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

                if user_answer.content.lower() == answer.lower():
                    await interaction.channel.send("Correct! Good job! 🎉")
                else:
                    await interaction.channel.send(f"Almost! The correct answer is: {answer.upper()}")
            except asyncio.TimeoutError:
                await interaction.channel.send("Time's up! Please try again.")
