import os
from openai import OpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_quotation(emotion):
    prompt = f"A quote about {emotion}:"

    response = client.completions.create(model="text-davinci-002",
    prompt=prompt,
    temperature=0.5,
    max_tokens=60)

    quotation = response.choices[0].text.strip()

    return quotation

# Test the function
print(generate_quotation("sadness"))
