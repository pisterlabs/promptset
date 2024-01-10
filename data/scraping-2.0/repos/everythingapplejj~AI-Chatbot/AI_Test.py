import openai
openai.api_key = "pk-McpgvqLGlIIuWxjqcbddjVnJiQArhovxRzdcmHwumehxcrYa"
openai.api_base = 'https://api.pawan.krd/v1'

while True:
    user = input("User:")
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt = user,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["Human: ", "AI: "]
    )
    text = response.choices[0].text
    print(response.choices[0].text)


