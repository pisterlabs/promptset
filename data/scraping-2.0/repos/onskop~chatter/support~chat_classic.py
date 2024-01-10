import openai

#open file .stranmlit/secrets.toml and read value of openai_key
with open('.streamlit/secrets.toml') as f:
    #search for line with openai_key
    openai_key = f.readline().split('=')[1].strip().replace('"', '')

print(openai_key)
openai.api_key = openai_key

with open('support/text.txt', encoding='utf-8') as f:
    prompt = f.read()

messages=[
        {"role": "system", "content": "You are a helpful assistant and answer with very long texts."},
        {"role": "user", "content": prompt}
    ]

for i in range(200):
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.1, max_tokens=150)
    print(response.choices[0].message.content)
    print('Run number: ', i, 'completed\n')
    print('-----------------------------------\n')