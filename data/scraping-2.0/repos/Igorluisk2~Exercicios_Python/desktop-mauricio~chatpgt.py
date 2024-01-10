import openai

openai.api_key = "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"
def get_compilation(prompt, model = "gpt-3.5-turbo"):
    messages=[{"role": "user","content":prompt}]
    response=openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message["content"]
while True:
    prompt=input("Digite sua pergunta: ")
    response=get_compilation(prompt)
    print(response)