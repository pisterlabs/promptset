import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_hyde_text(raw_text):
    # could use function calling here, but it's such a bloated abstraction that it's not worth it for a simple project like this
    
    # truncate raw_text to 2048 characters
    raw_text = raw_text[:2048]

    base_prompt = "Generate a specific description of this website based on its parsed text. Return nothing but the description."
    prompt = f"{base_prompt}\n\Website text:\n```{raw_text}```"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    return completion.choices[0].message.content.strip()