"""Run predictions based on the model

NOTE: This file is now depracated. Use src/inference/models.py instead
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# client = OpenAI()
client = None


def predict(model_name, prompt, model=None):
    if model_name == "gpt3.5":
        msgs = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=msgs, temperature=0
        )
        return response.choices[0].message.content

    # Use mistral model
    else:
        return model.predict(prompt)
