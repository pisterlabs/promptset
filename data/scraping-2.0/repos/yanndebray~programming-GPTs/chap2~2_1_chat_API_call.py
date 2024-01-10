import openai, tomli, json
# openai.api_key = "sk-XXXX"
with open('.streamlit/secrets.toml','rb') as f:
    secrets = tomli.load(f)
openai.api_key = secrets['OPENAI_API_KEY']
m = [{'role': 'system','content': 'If I say hello, say world'},
    {'role': 'user','content': 'hello'}]
completion = openai.chat.completions.create(model='gpt-3.5-turbo',
                                            messages=m)
response = completion.choices[0].message.content
print(response) # world