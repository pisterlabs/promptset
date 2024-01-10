import openai
openai.api_key = "API-Key"
output = openai.Completion.create(                 # using completion API
    prompt = "who am I?",
    model = "text-davinci-003"
)

"""
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_token=50,
    temperature=0;
    messages=[
        {"role"="system","content"="you are helful assistant"},
        {"role"="user", "content"=myinput}
    ]
)
print(response.choices[0].message.content)
"""

print(output["choices"][0]["text"])
