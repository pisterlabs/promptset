import os
import openai

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"


def generate_text(prompt):
    # Create an OpenAI client using your API key
    openai_client = openai.Client()

    # Generate text using the GPT-3 model
    response = openai_client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7,
    )

    # Extract the generated text from the response
    generated_text = response["choices"][0]["text"]

    return generated_text


prompt = "Write a haiku about a sunset."
generated_text = generate_text(prompt)
print(generated_text)
# import openai
# import requests
# from openai import OpenAI
# client = OpenAI()
# client = openai.Client(api_key=os.environ['OPENAI_API_KEY'],)
