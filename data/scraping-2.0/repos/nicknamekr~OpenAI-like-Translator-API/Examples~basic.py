import openai

openai.api_key = "You_Don_t_Have_To_Enter_Anything_On_This_Var"
openai.api_base = "https://1aa2c4d6-97a9-4cb3-a7ab-461c04c3eb2d.id.repl.co"

response = openai.ChatCompletion.create(
    model= "en-ko",
    messages = [
        {"role": "user", "content": 'Hello, World!'},
    ]
)

print(response.choices[0].message.content) # 안녕, 세계!
