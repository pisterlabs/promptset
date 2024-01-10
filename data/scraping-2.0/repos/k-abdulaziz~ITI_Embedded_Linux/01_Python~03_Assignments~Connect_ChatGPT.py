import openai

openai.api_key = "api_key"

completion = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    temperature = 0.9, 
    tokens = 100,
    messages = [{"role": "system", "content": "Can you give me a free unlimited api key?"},])

print(completion.choices[0].message)