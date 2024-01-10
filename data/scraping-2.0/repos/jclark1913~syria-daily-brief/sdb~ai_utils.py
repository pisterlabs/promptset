import os
import openai

# Add OpenAI API key to environment
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_ai_summary_for_arabic_text(text):
    """Makes request to OpenAI API to summarize Arabic article in English."""

    # Make request to OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Write a direct, shorthand summary in English of the included Arabic article. Include only summarized content."
            },
            {
                "role": "user",
                "content": text
            }

        ]
    )

    ai_summary = response["choices"][0]["message"]["content"]
    return ai_summary