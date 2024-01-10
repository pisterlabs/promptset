import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_code_from_caption(caption):
    prompt = f"Generate the code for the described React functional component. It should have React & CSS code too. Name the CSS file & import it. Do not include the image in the code {caption}"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1100,
    )
    generated_code = response['choices'][0]['text']
    return generated_code
