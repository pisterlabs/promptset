import os
import openai


def gptres(usersaid):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        return "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    if not usersaid or not usersaid.strip():
        return "Invalid input. Please provide a valid question or statement."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=usersaid,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )
    if response.choices[0].text:
        return response.choices[0].text.strip()
    else:
        return "Sorry, I couldn't understand your question or statement."
