import openai
from decouple import config

openai.organization = config("ORG_ID")
openai.api_key = config("OPENAI_API_KEY")


def generate_text(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, max_tokens=3000, n=1, stop=None, temperature=0.4
    )
    message = completions.choices[0].text
    return message.strip()


while True:
    print("-" * 50)
    # prompt = input('Faça sua pergunta para o ChatGPT:\n')
    prompt = "code bot discord python"
    print(f"R: {generate_text(prompt)}")
