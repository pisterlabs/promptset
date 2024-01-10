import openai

openai.api_key = "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"

def generate_prompt(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    return message.strip()
prompt = input('Faça sua pergunta: \n')
print(generate_prompt(prompt))